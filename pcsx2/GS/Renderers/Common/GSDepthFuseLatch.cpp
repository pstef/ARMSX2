// SPDX-FileCopyrightText: 2026 ARMSX2 Dev Team
// SPDX-License-Identifier: GPL-3.0+

#include "GS/Renderers/Common/GSDepthFuseLatch.h"

#include "common/Assertions.h"

#include <cstring>

GSDepthFuseLatch::GSDepthFuseLatch() = default;

GSDepthFuseLatch::~GSDepthFuseLatch() = default;

namespace
{
	/// The parts of the predicate both halves have to satisfy. Anything here either moves
	/// the draw's position in the stream from a detail to a semantic (a barrier, a feedback
	/// loop, colour clipping, destination alpha), splits it into more than one submission
	/// (a second pass, a drawlist), or lets the fragment shader decline to write a pixel -
	/// and a fold keeps one shader for both halves, so a discard in either one lands on the
	/// wrong set of pixels.
	bool IsFusableHalf(const GSHWDrawConfig& config)
	{
		if (config.require_one_barrier || config.require_full_barrier)
			return false;
		if (config.tex_hazard != GSHWDrawConfig::TEX_HAZARD_NONE)
			return false;
		if (config.ps.IsFeedbackLoopRT() || config.ps.IsFeedbackLoopDepth() || config.ps.tex_is_fb)
			return false;
		if (config.destination_alpha != GSHWDrawConfig::DestinationAlphaMode::Off)
			return false;
		if (config.depth.date || config.depth.date_one)
			return false;
		if (config.colclip_mode != GSHWDrawConfig::ColClipMode::NoModify)
			return false;
		if (config.alpha_test != GSHWDrawConfig::AlphaTestMode::NONE)
			return false;
		if (config.alpha_second_pass.enable || config.blend_multi_pass.enable)
			return false;
		if (config.drawlist || config.drawlist_bbox)
			return false;
		if (config.ps.HasShaderDiscard())
			return false;

		// Interlocked access orders the two halves against each other inside the shader, so
		// the pair is not two independent draws to begin with.
		if (config.ps.rov_color || config.ps.rov_depth != GSHWDrawConfig::PS_ROV_DEPTH::NONE)
			return false;

		// AA1 turns coverage into alpha and, on triangles, into a depth discard; it also
		// makes a line cover pixels twice where the expanded quads meet.
		if (config.ps.aa1 != GSHWDrawConfig::PS_AA1::NONE)
			return false;

		return true;
	}
} // namespace

bool GSDepthFuseLatch::IsHoldable(const GSHWDrawConfig& config)
{
	// Both attachments, colour written, depth tested but not written: the half the fold
	// keeps.
	if (!config.rt || !config.ds)
		return false;
	if (config.colormask.wrgba == 0 || config.ps.no_color)
		return false;
	if (config.depth.zwe)
		return false;
	if (config.depth.ztst == ZTST_ALWAYS)
		return false;

	// A masked write reads the target back, so the draw is not a plain write to it.
	if (config.ps.fbmask)
		return false;

	// Sampling an attachment of this very draw.
	if (config.tex == config.rt || config.tex == config.ds)
		return false;

	if (!IsFusableHalf(config))
		return false;

	// One primitive. With two, enabling depth writes lets the first one's stored depth fail
	// the second one's test, and the second one's colour - which the unfused pair wrote -
	// disappears.
	if (config.nverts == 0 || config.nindices == 0 || config.nindices != config.indices_per_prim)
		return false;
	if (config.nverts > MAX_VERTS || config.nindices > MAX_INDICES)
		return false;

	return true;
}

void GSDepthFuseLatch::Hold(const GSHWDrawConfig& config)
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

bool GSDepthFuseLatch::IsTwin(const GSHWDrawConfig& config) const
{
	pxAssert(m_held);

	// Writes depth into the held draw's depth buffer and nothing else.
	if (config.rt || config.colormask.wrgba != 0)
		return false;
	if (!config.ds || config.ds != m_config.ds)
		return false;
	if (!config.depth.zwe || config.depth.ztst != m_config.depth.ztst)
		return false;

	if (!IsFusableHalf(config))
		return false;

	// Sampling either attachment of the draw it is about to be folded into.
	if (config.tex == m_config.rt || config.tex == m_config.ds)
		return false;

	// Same geometry, and the same vertex shader to rasterize it with - the expansion mode
	// lives in vs, and the transform in cb_vs.
	if (config.topology != m_config.topology || config.indices_per_prim != m_config.indices_per_prim)
		return false;
	if (config.nverts != m_config.nverts || config.nindices != m_config.nindices)
		return false;
	if (config.vs.key != m_config.vs.key)
		return false;
	if (config.cb_vs != m_config.cb_vs)
		return false;
	if (std::memcmp(config.verts, m_config.verts, sizeof(GSVertex) * config.nverts) != 0)
		return false;
	if (std::memcmp(config.indices, m_config.indices, sizeof(u16) * config.nindices) != 0)
		return false;

	// Same depth value out of the fragment shader. zfloor quantizes it and zclamp caps it
	// against cb_ps.TA_MaxDepth_Af.z, and the renderer sets both from whether the draw
	// writes depth - so the two halves can disagree here even with identical geometry, and
	// the fold keeps the held draw's shader.
	if (config.ps.zfloor != m_config.ps.zfloor || config.ps.zclamp != m_config.ps.zclamp)
		return false;
	if (config.cb_ps.TA_MaxDepth_Af.z != m_config.cb_ps.TA_MaxDepth_Af.z)
		return false;

	// Same pixels reached.
	if (!config.scissor.eq(m_config.scissor) || !config.drawarea.eq(m_config.drawarea))
		return false;

	return true;
}

GSHWDrawConfig& GSDepthFuseLatch::Fuse()
{
	pxAssert(m_held);

	m_config.depth.zwe = true;

	return Release();
}

GSHWDrawConfig& GSDepthFuseLatch::Release()
{
	pxAssert(m_held);

	// Rewind the attachment state to what the draw would have found undeferred, so the
	// backend picks the same load op. It sets Dirty again itself.
	m_config.rt->SetStateForDeferral(m_rt_state);
	m_config.ds->SetStateForDeferral(m_ds_state);

	m_held = false;
	return m_config;
}

bool GSDepthFuseLatch::References(const GSTexture* tex) const
{
	if (!m_held || !tex)
		return false;

	return tex == m_config.rt || tex == m_config.ds || tex == m_config.tex || tex == m_config.pal;
}
