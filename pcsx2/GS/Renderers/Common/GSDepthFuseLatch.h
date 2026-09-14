// SPDX-FileCopyrightText: 2026 ARMSX2 Dev Team
// SPDX-License-Identifier: GPL-3.0+

#pragma once

#include "GS/Renderers/Common/GSDevice.h"

/// Holds one colour draw back for the length of a single RenderHW call, so that a
/// depth-only draw of the same geometry arriving immediately behind it can be folded into
/// it instead of submitted on its own.
///
/// The shape this exists for is a game that draws a primitive twice: once writing colour
/// with the depth buffer read-only, then again over the identical vertices writing nothing
/// but depth. Sly 3 draws a fifth of its frame that way, one two-vertex line at a time,
/// and each half costs a full draw's worth of renderer and driver work.
///
/// The fold is exact, and only because the two draws are constrained to the point where
/// it is: the colour draw writes no depth, so the depth draw's test reads exactly the
/// buffer the colour draw's test read; the depth draw writes no colour, so nothing it does
/// is lost by dropping it. Turning depth writes on in the colour draw therefore stores
/// what the depth draw would have stored, at the pixels the depth draw would have stored
/// it.
///
/// The trap is that the two draws do not have to agree about the depth *value*: the
/// renderer decides ps.zfloor and ps.zclamp from whether the draw writes depth, so the
/// colour half can quantize its depth differently from the depth half, and whichever
/// shader survives the fold is the one whose gl_FragDepth reaches the buffer. So the twin
/// test compares those bits and the constants behind them, rather than assuming identical
/// geometry implies an identical depth.
///
/// The second trap is fragment discard. A fold keeps the colour draw's pixel shader, so a
/// pixel that shader kills stores no depth, and a pixel the depth draw's shader would have
/// killed stores depth anyway. Both halves are required to discard nothing.
///
/// The held draw is the newest one in flight, so it is submitted behind whatever
/// GSPassScheduler already has queued, and every entry point that drains the scheduler
/// drains this too - the latch is reached through the same GSDevice::FlushDeferredDraws()
/// and FlushDeferredDrawsFor() wrappers, and names its textures through the same
/// DeferredDrawsReference(), so a held config can never outlive a GSTexture it points at.
class GSDepthFuseLatch
{
public:
	/// The mechanism is aimed at the per-draw cost of tiny draws, and the hold copies the
	/// geometry, so the copy is kept to something a few stores can move.
	static constexpr u32 MAX_VERTS = 8;
	static constexpr u32 MAX_INDICES = 8;

	GSDepthFuseLatch();
	~GSDepthFuseLatch();

	__fi bool IsHeld() const { return m_held; }

	/// True when a draw is worth holding back: it writes colour, reads depth without
	/// writing it, is one small primitive, and carries nothing that makes its position in
	/// the stream or its fragment coverage load-bearing.
	static bool IsHoldable(const GSHWDrawConfig& config);

	/// Takes the draw, geometry included. The attachments are moved to Dirty for the
	/// duration, the way an undeferred submission would have left them, and put back at
	/// Release().
	void Hold(const GSHWDrawConfig& config);

	/// True when [config] is the held draw's depth-only twin: same geometry, same depth
	/// test, writes depth and no colour, and agrees with the held draw about what depth
	/// value the fragment shader produces.
	bool IsTwin(const GSHWDrawConfig& config) const;

	/// Turns the held draw into the fused draw and hands it back, emptying the latch. The
	/// reference is into the latch and stays valid until the next Hold().
	GSHWDrawConfig& Fuse();

	/// Hands the held draw back unchanged, emptying the latch.
	GSHWDrawConfig& Release();

	/// True if the held draw reads or writes this texture.
	bool References(const GSTexture* tex) const;

	/// Drops the held draw without rendering it, and without putting the attachment state
	/// back. Only for device teardown, where the targets are going away anyway.
	void Clear() { m_held = false; }

private:
	/// verts/indices are repointed at the arrays below in Hold().
	GSHWDrawConfig m_config = {};
	GSVertex m_verts[MAX_VERTS] = {};
	u16 m_indices[MAX_INDICES] = {};

	/// What the attachments' State was when the draw was held; see GSPassScheduler for why
	/// the deferral window has to be invisible to the texture cache.
	GSTexture::State m_rt_state = GSTexture::State::Dirty;
	GSTexture::State m_ds_state = GSTexture::State::Dirty;

	bool m_held = false;
};
