// SPDX-FileCopyrightText: 2026 ARMSX2 Dev Team
// SPDX-License-Identifier: GPL-3.0+

#include "GS/Renderers/Common/GSDualFuseLatch.h"

#include "GS/Renderers/Common/GSDepthFuseLatch.h"
#include "GS/GSPerfMon.h"

#include "common/Assertions.h"

#include <cstring>

GSDualFuseLatch::GSDualFuseLatch() = default;

GSDualFuseLatch::~GSDualFuseLatch() = default;

namespace
{
	/// The selector bits that describe one half alone. Every other bit has to match across the
	/// pair, because one shader runs both stages off it.
	///
	/// blend_a..d: the base half has no blend arm at all and the half over it has the
	/// accumulation one, which IsTwin() pins to its exact values.
	/// fba, rta_correction: the byte the fused draw stores in alpha is the second stage's, so
	/// its alpha fixup and its output scale are the ones that run. Carried in dual_fba and
	/// dual_rta_correction.
	/// no_color1: the second output belongs to whichever half had dual-source blending. The
	/// fused draw has blending off and emits none.
	/// zfloor: the renderer sets it from whether a half writes depth, so the two halves
	/// disagree by construction. The fused draw keeps the base half's, which is the depth it
	/// writes.
	/// dual, dual_fba, dual_rta_correction: not set on either half, written by the fuse.
	const GSHWDrawConfig::PSSelector& DualExemptMask()
	{
		static const GSHWDrawConfig::PSSelector mask = []() {
			GSHWDrawConfig::PSSelector m;
			m.blend_a = m.blend_b = m.blend_c = m.blend_d = 3;
			m.fba = 1;
			m.rta_correction = 1;
			m.no_color1 = 1;
			m.zfloor = 1;
			m.dual = m.dual_fba = m.dual_rta_correction = 1;
			return m;
		}();
		return mask;
	}

	/// True when the PS_DUAL body reproduces what this selector asks for. It runs the texture
	/// fetch, the TFX combine and the fog a second time and then one fixed blend arm; every
	/// selector that puts work anywhere else in the fragment program - a destination read, a
	/// discard, a shuffle, a dither, a channel fetch, a second output - is refused rather than
	/// written twice.
	bool DualStageIsReproducible(const GSHWDrawConfig::PSSelector& ps)
	{
		// Already fused.
		if (ps.dual)
			return false;

		// Colour comes out of a sampled texture, not a palette lookup, a depth buffer or a
		// channel fetch of the render target.
		if (ps.pal_fmt || ps.depth_fmt || ps.channel)
			return false;

		// Shuffles rewrite the colour as packed 16-bit fields, which is not a second colour to
		// add but a different destination layout.
		if (ps.shuffle || ps.shuffle_same || ps.real16src || ps.process_ba || ps.process_rg ||
			ps.shuffle_across || ps.write_rg)
			return false;

		// All three read or rewrite the stored byte rather than producing a colour to add.
		if (ps.fbmask || ps.quantize_color || ps.substitute_alpha)
			return false;

		// Dither is added before the clamp, so it belongs to a single colour, not to a sum.
		if (ps.dither || ps.dither_adjust)
			return false;

		// Colour clipping renormalises the target to 16 bits per channel and wraps instead of
		// clamping, so the one store the fuse leaves is not the store either half made.
		if (ps.colclip || ps.colclip_hw)
			return false;

		// Everything here either moves work into the blend unit, where the fuse has no second
		// slot for it, or rewrites the alpha the stages are combined through.
		if (ps.blend_hw || ps.blend_mix || ps.a_masked || ps.round_inv || ps.pabe || ps.fixed_one_a ||
			ps.blend_factor_in_alpha || ps.af_in_src1)
			return false;

		// Anything that can decline to write the pixel lands on both stages or neither.
		if (ps.date || ps.ztst || ps.scanmsk || ps.no_color)
			return false;
		if (ps.atst != GSHWDrawConfig::PS_ATST::NONE || ps.afail != GSHWDrawConfig::PS_AFAIL::KEEP)
			return false;
		if (ps.aa1 != GSHWDrawConfig::PS_AA1::NONE)
			return false;
		if (ps.rov_color || ps.rov_depth != GSHWDrawConfig::PS_ROV_DEPTH::NONE)
			return false;

		// The anisotropic road is a sampling loop per fragment, and a fused draw would run one
		// for each stage.
		if (ps.sw_aniso > 1)
			return false;

		// Samples the render target rather than the stage's own texture.
		if (ps.tex_is_fb)
			return false;

		return true;
	}

	/// The geometry the pair has to share: everything a GSVertex carries except the colour,
	/// which is the one thing the second stage brings of its own. RGBA sits at offset 8 and Q
	/// at 12, so the two runs are [0,8) and [12,32).
} // namespace

bool GSDualFuseLatch::IsHoldable(const GSHWDrawConfig& config)
{
	// Both attachments, every colour channel written, and no blending of its own: the fused
	// shader adds the two stages itself and leaves the blend unit idle, so the base half's
	// colour has to be finished inside its own shader.
	if (!config.rt || !config.ds)
		return false;
	if (config.colormask.wrgba != 0xf)
		return false;
	if (config.blend.enable)
		return false;
	if (config.ps.blend_a || config.ps.blend_b || config.ps.blend_c || config.ps.blend_d)
		return false;
	if (!config.ps.no_color1)
		return false;

	// The depth test the fused draw keeps. The half over it tests ZTST_GEQUAL, which is the
	// pair's only inexactness; see the class comment.
	if (config.depth.ztst != ZTST_GREATER)
		return false;

	// A texture that is neither attachment, with the CLUT already expanded into it - the
	// second stage samples its own texture through the same sampler, and there is one palette
	// binding.
	if (!config.tex || config.tex == config.rt || config.tex == config.ds)
		return false;
	if (config.pal)
		return false;

	// Indexed triangles read straight out of the vertex buffer: the fused draw interleaves the
	// two halves' vertices at twice the stride, which a vertex shader that expands its input
	// out of a storage buffer would not see.
	if (config.topology != GSHWDrawConfig::Topology::Triangle || config.indices_per_prim != 3)
		return false;
	if (config.vs.expand != GSHWDrawConfig::VSExpand::None)
		return false;

	if (!GSIsFusableDrawHalf(config))
		return false;
	if (!DualStageIsReproducible(config.ps))
		return false;

	if (config.nverts == 0 || config.nindices == 0)
		return false;
	if (config.nverts > MAX_VERTS || config.nindices > MAX_INDICES)
		return false;

	return true;
}

void GSDualFuseLatch::Hold(const GSHWDrawConfig& config)
{
	pxAssert(!m_held && IsHoldable(config));

	m_config = config;

	// GSState reuses its vertex and index buffers for the very next draw, so the geometry
	// has to be taken by value.
	std::memcpy(m_verts, config.verts, sizeof(GSVertex) * config.nverts);
	std::memcpy(m_indices, config.indices, sizeof(u16) * config.nindices);
	m_config.verts = m_verts;
	m_config.indices = m_indices;

	// An undeferred submission would have bound both attachments and flipped them to Dirty
	// on the spot. The texture cache reads GSTexture::State directly, without going through
	// an entry point that could flush, so the hold has to do the same and put the original
	// back at Release() for the backend to pick its load op from.
	m_rt_state = m_config.rt->GetState();
	m_config.rt->SetStateForDeferral(GSTexture::State::Dirty);
	m_ds_state = m_config.ds->GetState();
	m_config.ds->SetStateForDeferral(GSTexture::State::Dirty);

	m_held = true;
}

bool GSDualFuseLatch::IsTwin(const GSHWDrawConfig& config) const
{
	pxAssert(m_held);

	// Same framebuffer, every colour channel written, depth tested against what the held draw
	// is about to write but not written itself.
	if (config.rt != m_config.rt || config.ds != m_config.ds)
		return false;
	if (config.colormask.wrgba != 0xf)
		return false;
	if (config.depth.zwe || config.depth.ztst != ZTST_GEQUAL)
		return false;

	// The accumulation this fuse implements, and only it: the shader hands the blend unit
	// trunc(Cs*As) and it adds that to the target with alpha replaced. The fused shader adds
	// the same term to the base stage's colour, so any other factor, operator or constant is a
	// different arithmetic.
	if (!config.blend.enable || config.blend.constant_enable)
		return false;
	if (config.blend.src_factor != GSDevice::CONST_ONE || config.blend.dst_factor != GSDevice::CONST_ONE)
		return false;
	if (config.blend.op != GSDevice::OP_ADD)
		return false;
	if (config.blend.src_factor_alpha != GSDevice::CONST_ONE ||
		config.blend.dst_factor_alpha != GSDevice::CONST_ZERO)
		return false;

	// (Cs - 0) * As + 0, the shader half of the same arithmetic. blend_c is pinned to As
	// rather than Af because the fused body multiplies by the second stage's own alpha.
	if (config.ps.blend_a != 0 || config.ps.blend_b != 2 || config.ps.blend_c != 0 || config.ps.blend_d != 2)
		return false;

	if (!GSIsFusableDrawHalf(config))
		return false;
	if (!DualStageIsReproducible(config.ps))
		return false;

	if (!config.tex || config.tex == config.rt || config.tex == config.ds)
		return false;
	if (config.pal)
		return false;

	// Same rasterization: the fused draw issues one vertex shader over one index list.
	if (config.topology != m_config.topology || config.indices_per_prim != m_config.indices_per_prim)
		return false;
	if (config.nverts != m_config.nverts || config.nindices != m_config.nindices)
		return false;
	if (config.vs.key != m_config.vs.key)
		return false;
	if (config.cb_vs != m_config.cb_vs)
		return false;
	if (!config.scissor.eq(m_config.scissor) || !config.drawarea.eq(m_config.drawarea))
		return false;

	// One sampler binding serves both textures.
	if (config.sampler.key != m_config.sampler.key)
		return false;

	// Same fragment program outside the bits the second stage carries for itself.
	const GSHWDrawConfig::PSSelector& mask = DualExemptMask();
	if (((config.ps.key_lo ^ m_config.ps.key_lo) & ~mask.key_lo) != 0)
		return false;
	if (((config.ps.key_hi ^ m_config.ps.key_hi) & ~mask.key_hi) != 0)
		return false;

	// Same constants outside FogColor_AREF, which the fuse parks in DitherMatrix[0]. It
	// leads the block, so the rest is one compare.
	static_assert(offsetof(GSHWDrawConfig::PSConstantBuffer, FogColor_AREF) == 0);
	if (std::memcmp(reinterpret_cast<const u8*>(&config.cb_ps) + sizeof(GSVector4),
			reinterpret_cast<const u8*>(&m_config.cb_ps) + sizeof(GSVector4),
			sizeof(GSHWDrawConfig::PSConstantBuffer) - sizeof(GSVector4)) != 0)
		return false;

	// Same geometry, down to the byte outside the vertex colour.
	if (std::memcmp(config.indices, m_config.indices, sizeof(u16) * config.nindices) != 0)
		return false;
	return SameGeometryOutsideColour(config.verts, m_config.verts, config.nverts);
}

bool GSDualFuseLatch::SameGeometryOutsideColour(const GSVertex* a, const GSVertex* b, u32 count)
{
	// The differences are or'd across the vertices and tested once, so the loop carries no
	// branch.
	static_assert(offsetof(GSVertex, RGBAQ) == 8);
	const GSVector4i colour_mask = GSVector4i(-1, -1, 0, -1);
	GSVector4i diff = GSVector4i::zero();
	for (u32 i = 0; i < count; i++)
	{
		diff |= (GSVector4i(a[i].m[0]) ^ GSVector4i(b[i].m[0])) & colour_mask;
		diff |= GSVector4i(a[i].m[1]) ^ GSVector4i(b[i].m[1]);
	}
	return diff.eq(GSVector4i::zero());
}

GSHWDrawConfig& GSDualFuseLatch::Fuse(const GSHWDrawConfig& twin)
{
	pxAssert(m_held && IsTwin(twin));

	// The vertex input walks one binding at twice the stride, so the two halves' vertices go
	// in alternating slots. Only the colour of the odd ones is read, but writing the whole
	// vertex keeps the array one shape.
	for (u32 i = 0; i < m_config.nverts; i++)
	{
		m_pair_verts[i * 2] = m_verts[i];
		m_pair_verts[i * 2 + 1] = twin.verts[i];
	}
	m_config.verts = m_pair_verts;

	m_config.ps.dual = 1;
	m_config.ps.dual_fba = twin.ps.fba;
	m_config.ps.dual_rta_correction = twin.ps.rta_correction;
	m_config.vs.dual = 1;
	m_config.dual_tex = twin.tex;
	// The dither matrix is free: both halves are refused if they dither. It carries the
	// second layer's fog colour so the constant block does not grow for one vector.
	m_config.cb_ps.DitherMatrix[0] = twin.cb_ps.FogColor_AREF;

	g_perfmon.Put(GSPerfMon::DualFusedPairs, 1);

	return Release();
}

GSHWDrawConfig& GSDualFuseLatch::Release()
{
	pxAssert(m_held);

	// Rewind the attachment state to what the draw would have found undeferred, so the
	// backend picks the same load op. It sets Dirty again itself.
	m_config.rt->SetStateForDeferral(m_rt_state);
	m_config.ds->SetStateForDeferral(m_ds_state);

	m_held = false;
	return m_config;
}

bool GSDualFuseLatch::References(const GSTexture* tex) const
{
	if (!m_held || !tex)
		return false;

	return tex == m_config.rt || tex == m_config.ds || tex == m_config.tex || tex == m_config.pal ||
		   tex == m_config.dual_tex;
}
