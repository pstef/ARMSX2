// SPDX-FileCopyrightText: 2026 ARMSX2 Dev Team
// SPDX-License-Identifier: GPL-3.0+

// GSVertexKernels::ParsePlannedVertex (GS/GSVertexKick.h): the vertex build for
// a shape GIFBuildShapePlan described, with no part of the shape known at
// compile time.
//
// Checked both ways round. Against the seven layouts already shipped: for each
// of them there is a plan vertex that says the same thing, and the two parses
// have to produce the same 32 bytes. And against a scalar model written straight
// off the packed register formats -- integer field extraction, no vectors, no
// branch on which position descriptor it is -- over random shapes the shipped
// layouts do not cover, which is where a UV beside an XYZF2, a fog descriptor,
// or four vertices to a period get looked at.

#include <gtest/gtest.h>

#include <cmath>
#include <cstring>
#include <random>
#include <vector>

#include "GS/GSVertexKick.h"

namespace
{
	constexpr u8 kC = GIF_SHAPE_CARRIED;

	GIFShapeVertex MakeVertex(u8 st, u8 q, u8 rgba, u8 uv, u8 fog, u8 off, u8 desc)
	{
		GIFShapeVertex v = {};
		v.src = {st, q, rgba, uv, fog};
		v.off = off;
		v.desc = desc;
		return v;
	}

	// The Q fix-ups GIFPackedRegHandlerSTQ applies before a colour write copies
	// the latch into the vertex.
	u32 FixupQ(u32 q)
	{
		if (q == 0)
			q = 0x00800000u; // FLT_MIN
		float f;
		std::memcpy(&f, &q, sizeof(f));
		if (std::isnan(f))
			q = 0x7F7FFFFFu; // FLT_MAX
		return q;
	}

	struct ModelVertex
	{
		u32 m0[4];
		u32 m1[4];
	};

	// The vertex the packed register formats say the plan builds. Every field is
	// pulled out of the qword the plan names by shift and mask.
	ModelVertex Model(const GIFPackedReg* base, const GIFShapeVertex& v, const u32 carry_m0[4],
		u32 latched_q, u64 carry_uvfog)
	{
		ModelVertex e = {};

		if (v.src.st != kC)
		{
			e.m0[0] = base[v.src.st].U32[0];
			e.m0[1] = base[v.src.st].U32[1];
		}
		else
		{
			e.m0[0] = carry_m0[0];
			e.m0[1] = carry_m0[1];
		}

		if (v.src.rgba != kC)
		{
			const GIFPackedReg& r = base[v.src.rgba];
			e.m0[2] = (r.U32[0] & 0xFF) | ((r.U32[1] & 0xFF) << 8) | ((r.U32[2] & 0xFF) << 16) |
			          ((r.U32[3] & 0xFF) << 24);
			e.m0[3] = (v.src.q != kC) ? FixupQ(base[v.src.q].U32[2]) : latched_q;
		}
		else
		{
			e.m0[2] = carry_m0[2];
			e.m0[3] = carry_m0[3];
		}

		u32 uv = static_cast<u32>(carry_uvfog);
		if (v.src.uv != kC)
			uv = (base[v.src.uv].U32[0] & 0x3FFF) | ((base[v.src.uv].U32[1] & 0x3FFF) << 16);

		// A FOG descriptor and an XYZF2 carry F in the same bits, so one
		// extraction covers a vertex that reads its own position qword and one
		// that reads a fog write.
		u32 fog = static_cast<u32>(carry_uvfog >> 32);
		if (v.src.fog != kC)
			fog = (base[v.src.fog].U32[3] >> 4) & 0xFF;

		const GIFPackedReg& p = base[v.off];
		e.m1[0] = (p.U32[0] & 0xFFFF) | ((p.U32[1] & 0xFFFF) << 16);
		e.m1[1] = v.IsXYZF() ? ((p.U32[2] >> 4) & 0x00FFFFFF) : p.U32[2];
		e.m1[2] = uv;
		e.m1[3] = fog;
		return e;
	}

	// Bit patterns a Q, a colour byte or a coordinate is worth being given.
	const u32 kEdgeWords[] = {
		0x00000000u, // +0.0, which the Q fix-up rewrites
		0x80000000u, // -0.0, which it does not
		0x7F800000u, // +inf
		0xFF800000u, // -inf
		0x7FC00000u, // quiet NaN
		0x7F800001u, // signalling NaN
		0xFFFFFFFFu,
		0x00000001u, // denormal
		0x00800000u, // FLT_MIN
		0x7F7FFFFFu, // FLT_MAX
		0x3F800000u, // 1.0
		0x0000FFFFu,
		0xFFFF0000u,
		0x00008000u, // the ADC bit in a position's fourth word
	};

	u32 RandomWord(std::mt19937& rng)
	{
		const u32 roll = rng() % 4;
		if (roll == 0)
			return kEdgeWords[rng() % std::size(kEdgeWords)];
		return static_cast<u32>(rng());
	}

	void FillRecords(std::mt19937& rng, GIFPackedReg* r, u32 count)
	{
		for (u32 i = 0; i < count; i++)
			for (u32 w = 0; w < 4; w++)
				r[i].U32[w] = RandomWord(rng);
	}

	::testing::AssertionResult SameVertex(const GSVector4i& m0, const GSVector4i& m1,
		const u32 want_m0[4], const u32 want_m1[4])
	{
		for (u32 i = 0; i < 4; i++)
		{
			if (m0.U32[i] != want_m0[i])
				return ::testing::AssertionFailure()
				       << "m[0] lane " << i << ": got " << std::hex << m0.U32[i] << " want " << want_m0[i];
			if (m1.U32[i] != want_m1[i])
				return ::testing::AssertionFailure()
				       << "m[1] lane " << i << ": got " << std::hex << m1.U32[i] << " want " << want_m1[i];
		}
		return ::testing::AssertionSuccess();
	}
} // namespace

// ---------------------------------------------------------------------------
// Against the layouts already shipped.
// ---------------------------------------------------------------------------

namespace
{
	using GSVertexKernels::PackedLayout;

	// One shipped layout, the plan vertex that says the same thing, and enough
	// qwords to hold it.
	struct LayoutCase
	{
		const char* name;
		PackedLayout layout;
		GIFPackedLayout off;
		GIFShapeVertex plan;
		// The two contiguous triples do not apply the NaN half of the Q fix-up;
		// that divergence is inherited from upstream and pinned where it stands,
		// so those two are only compared on a Q that is not a NaN.
		bool skips_the_nan_fixup;
	};

	template <PackedLayout L>
	void RunLayoutCase(const LayoutCase& c)
	{
		std::mt19937 rng(0xB1A5u + static_cast<u32>(L));
		GIFPackedReg r[8];

		for (int iter = 0; iter < 4000; iter++)
		{
			FillRecords(rng, r, c.off.stride);

			GSVector4i carry_m0;
			for (u32 i = 0; i < 4; i++)
				carry_m0.U32[i] = RandomWord(rng);
			const u32 q_word = RandomWord(rng);
			float q_latch;
			std::memcpy(&q_latch, &q_word, sizeof(q_latch));
			const u64 uvfog = (static_cast<u64>(RandomWord(rng)) << 32) | RandomWord(rng);

			if (c.skips_the_nan_fixup && std::isnan(*reinterpret_cast<const float*>(&r[c.off.off_a].U32[2])))
				continue;

			GSVector4i want0, want1;
			GSVertexKernels::ParsePackedRecord<L>(r, c.off, uvfog,
				GSVertexKernels::MakeLayoutCarry(L, carry_m0, q_latch), want0, want1);

			GSVertexKernels::PlanCarry carry;
			carry.m0 = carry_m0;
			carry.q = GSVector4i::load(static_cast<int>(q_word));
			carry.uvfog = uvfog;

			GSVector4i got0, got1;
			GSVertexKernels::ParsePlannedVertex(r, c.plan, carry, got0, got1);

			ASSERT_TRUE(SameVertex(got0, got1, want0.U32, want1.U32))
				<< c.name << " iteration " << iter;
		}
	}
} // namespace

TEST(PlannedParse, MatchesEveryShippedLayout)
{
	// {ST, RGBAQ, XYZF2} contiguous. The position qword is the fog source: an
	// XYZF2 writes the fog register with its own F.
	const LayoutCase triple_f = {"TripleXYZF2", PackedLayout::TripleXYZF2, {3, 0, 1, 2},
		MakeVertex(0, 0, 1, kC, 2, 2, GIF_REG_XYZF2), true};
	const LayoutCase triple_z = {"TripleXYZ2", PackedLayout::TripleXYZ2, {3, 0, 1, 2},
		MakeVertex(0, 0, 1, kC, kC, 2, GIF_REG_XYZ2), true};
	const LayoutCase nop_triple = {"NopTripleXYZF2", PackedLayout::NopTripleXYZF2, {5, 1, 3, 4},
		MakeVertex(1, 1, 3, kC, 4, 4, GIF_REG_XYZF2), false};
	const LayoutCase pair_stq = {"PairSTQXYZ2", PackedLayout::PairSTQXYZ2, {3, 0, 0, 2},
		MakeVertex(0, kC, kC, kC, kC, 2, GIF_REG_XYZ2), false};
	const LayoutCase pair_uv = {"PairUVXYZ2", PackedLayout::PairUVXYZ2, {4, 1, 0, 3},
		MakeVertex(kC, kC, kC, 1, kC, 3, GIF_REG_XYZ2), false};
	const LayoutCase pair_rgba = {"PairRGBAQXYZ2", PackedLayout::PairRGBAQXYZ2, {2, 0, 0, 1},
		MakeVertex(kC, kC, 0, kC, kC, 1, GIF_REG_XYZ2), false};
	const LayoutCase single = {"SingleXYZF2", PackedLayout::SingleXYZF2, {4, 0, 0, 3},
		MakeVertex(kC, kC, kC, kC, 3, 3, GIF_REG_XYZF2), false};

	RunLayoutCase<PackedLayout::TripleXYZF2>(triple_f);
	RunLayoutCase<PackedLayout::TripleXYZ2>(triple_z);
	RunLayoutCase<PackedLayout::NopTripleXYZF2>(nop_triple);
	RunLayoutCase<PackedLayout::PairSTQXYZ2>(pair_stq);
	RunLayoutCase<PackedLayout::PairUVXYZ2>(pair_uv);
	RunLayoutCase<PackedLayout::PairRGBAQXYZ2>(pair_rgba);
	RunLayoutCase<PackedLayout::SingleXYZF2>(single);
}

// ---------------------------------------------------------------------------
// Against the register formats.
// ---------------------------------------------------------------------------

TEST(PlannedParse, MatchesTheScalarModel)
{
	// Every descriptor a plan can hold, so the shapes drawn here run past what
	// any shipped layout covers: several vertices to a period, a UV beside an
	// XYZF2, a fog descriptor, a colour with no ST before it.
	const u8 alphabet[] = {GIF_REG_RGBA, GIF_REG_STQ, GIF_REG_UV, GIF_REG_XYZF2, GIF_REG_XYZ2,
		GIF_REG_FOG, GIF_REG_XYZF3, GIF_REG_XYZ3, GIF_REG_NOP};

	std::mt19937 rng(0x9E3Du);
	u32 planned = 0, vertices = 0;

	for (int iter = 0; iter < 40000; iter++)
	{
		const u32 nreg = 1 + rng() % 16;

		GSVector4i regs = GSVector4i::zero();
		for (u32 i = 0; i < nreg; i++)
			regs.U8[i] = alphabet[rng() % std::size(alphabet)];

		GIFShapePlan plan = {};
		if (!GIFBuildShapePlan(regs, nreg, plan))
			continue;
		planned++;

		GIFPackedReg r[16];
		FillRecords(rng, r, nreg);

		u32 carry_m0[4];
		for (u32& w : carry_m0)
			w = RandomWord(rng);
		const u32 q_word = RandomWord(rng);
		const u64 uvfog = (static_cast<u64>(RandomWord(rng)) << 32) | RandomWord(rng);

		GSVertexKernels::PlanCarry carry;
		std::memcpy(&carry.m0, carry_m0, sizeof(carry_m0));
		carry.q = GSVector4i::load(static_cast<int>(q_word));
		carry.uvfog = uvfog;

		for (u32 j = 0; j < plan.count; j++)
		{
			const ModelVertex want = Model(r, plan.vertices[j], carry_m0, q_word, uvfog);

			GSVector4i got0, got1;
			GSVertexKernels::ParsePlannedVertex(r, plan.vertices[j], carry, got0, got1);

			ASSERT_TRUE(SameVertex(got0, got1, want.m0, want.m1))
				<< "iteration " << iter << " vertex " << j << " of " << static_cast<int>(plan.count);
			vertices++;
		}
	}

	// A liveness clause for the loop above: a filter that stopped accepting
	// shapes would leave every assertion unreached.
	EXPECT_GT(planned, 5000u);
	EXPECT_GT(vertices, 10000u);
}
