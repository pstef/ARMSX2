// SPDX-FileCopyrightText: 2026 ARMSX2 Dev Team
// SPDX-License-Identifier: GPL-3.0+

#pragma once

#include "GS/Renderers/Common/GSDevice.h"

/// Holds one textured colour draw back for the length of a single RenderHW call, so that a
/// second textured colour draw over the same geometry arriving immediately behind it can be
/// folded into it as a second shader stage instead of a second submission.
///
/// The pair: a base layer drawn with one palette and one set of vertex colours, and over it the
/// same mesh - the same positions, Z, texture coordinates and index list - with a second palette
/// of the same texture, a second set of vertex colours, and an accumulation blend that adds its
/// result to what the base layer just wrote.
///
/// The fused draw keeps the base layer's depth state, blending off, and a fragment shader that
/// computes both layers' colours, applies the second layer's blend arm to the second result and
/// stores the sum. Both terms are integer-valued and the base layer's is already inside
/// [0, 255], so the single UNORM8 store clamps exactly where the second of the two original
/// stores did. Alpha comes from the second layer, which is what the pair left in the target.
///
/// What the fold does not reproduce: the two layers do not use the same depth comparison. The
/// base layer tests ZTST_GREATER and writes depth; the layer over it tests ZTST_GEQUAL and
/// writes none, so at a fragment whose depth equals the stored depth the pair draws the second
/// layer alone - the base layer fails, and the layer over it passes against the unchanged
/// buffer. One draw carrying the base layer's comparison draws neither. On this device no
/// fragment shader can read the stored depth (no texture barrier, no framebuffer fetch, no ROV,
/// and depth targets are not allocated as input attachments), so the fused draw cannot gate its
/// two stages separately. The fragments that differ are exactly those whose depth equals the
/// buffer's.
///
/// The held draw is the newest one in flight, so it is submitted behind whatever
/// GSPassScheduler already has queued, and every entry point that drains the scheduler drains
/// this too - the latch is reached through the same GSDevice::FlushDeferredDraws() and
/// FlushDeferredDrawsFor() wrappers, and names its textures through the same
/// DeferredDrawsReference(), so a held config can never outlive a GSTexture it points at.
class GSDualFuseLatch
{
public:
	/// Bounds both copies of the geometry: the hold takes the base half's, and the fuse builds
	/// the interleaved pair out of it. A draw above either cap is not held.
	static constexpr u32 MAX_VERTS = 128;
	static constexpr u32 MAX_INDICES = 384;

	GSDualFuseLatch();
	~GSDualFuseLatch();

	__fi bool IsHeld() const { return m_held; }

	/// The held draw. Only meaningful while IsHeld().
	__fi const GSHWDrawConfig& Held() const { pxAssert(m_held); return m_config; }

	/// True when the two vertex arrays are equal outside their colour: the geometry half of
	/// IsTwin(), exposed so the renderer can ask it of raw vertices before a config exists.
	static bool SameGeometryOutsideColour(const GSVertex* a, const GSVertex* b, u32 count);

	/// True when a draw is worth holding back as the base layer of a pair: it writes colour
	/// through every channel with no blending of its own, tests depth with ZTST_GREATER, samples
	/// a texture that is neither attachment, and carries nothing the fused shader cannot
	/// reproduce.
	static bool IsHoldable(const GSHWDrawConfig& config);

	/// Takes the draw, geometry included. The attachments are moved to Dirty for the duration,
	/// the way an undeferred submission would have left them, and put back at Release().
	void Hold(const GSHWDrawConfig& config);

	/// True when [config] is the held draw's accumulation twin: the same geometry down to the
	/// byte outside the vertex colour (read here, unless the renderer set
	/// geometry_matches_held), the same shader configuration outside the bits the second
	/// stage carries for itself, ZTST_GEQUAL with no depth write, and the ONE/ONE accumulation
	/// of a shader-side Cs*As.
	bool IsTwin(const GSHWDrawConfig& config) const;

	/// Turns the held draw into the fused draw and hands it back, emptying the latch. [twin]
	/// must be the config IsTwin() accepted. The reference is into the latch and stays valid
	/// until the next Hold().
	GSHWDrawConfig& Fuse(const GSHWDrawConfig& twin);

	/// Hands the held draw back unchanged, emptying the latch.
	GSHWDrawConfig& Release();

	/// True if the held draw reads or writes this texture.
	bool References(const GSTexture* tex) const;

private:
	/// verts/indices are repointed at the arrays below in Hold(), and at m_pair_verts in
	/// Fuse().
	GSHWDrawConfig m_config = {};
	GSVertex m_verts[MAX_VERTS] = {};
	u16 m_indices[MAX_INDICES] = {};

	/// The fused draw's vertex array: the two halves' vertices interleaved, base at the even
	/// slots and twin at the odd ones, submitted at twice the GSVertex stride. Built in Fuse(),
	/// so a hold that does not find its twin never pays for it.
	GSVertex m_pair_verts[MAX_VERTS * 2] = {};

	/// What the attachments' State was when the draw was held; see GSPassScheduler for why
	/// the deferral window has to be invisible to the texture cache.
	GSTexture::State m_rt_state = GSTexture::State::Dirty;
	GSTexture::State m_ds_state = GSTexture::State::Dirty;

	bool m_held = false;
};
