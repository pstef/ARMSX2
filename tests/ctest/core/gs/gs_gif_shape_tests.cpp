// SPDX-FileCopyrightText: 2026 ARMSX2 Dev Team
// SPDX-License-Identifier: GPL-3.0+

// GIFBuildShapePlan (GS/GSRegs.h): a packed tag's descriptor list reduced to
// where every field of every vertex it builds is read from.
//
// Two things are checked here, and one deliberately is not.
//
// Against the classifier already in the tree: every layout GIFPath::SetTag
// recognises has to come back out of the plan with the offsets SetTag wrote into
// GIFPackedLayout. The two describe the same tags by different routes, so a
// disagreement means one of them is wrong.
//
// Against a flat walk of four unrolled periods: that walk tracks, per field, the
// absolute qword that last wrote it, and has no notion of a period at all. It
// decides independently whether the plan's one-period answer holds from period 1
// on, and whether a rejected tag really did read a field the period before it
// wrote.
//
// What the flat walk does not check is what a descriptor means -- it encodes the
// same per-descriptor rules the plan does, so the two agree there by
// construction rather than by test. Those rules are pinned below one descriptor
// value at a time; running them against the per-qword handlers is a differential
// test on a stream, not on a tag.

#include <gtest/gtest.h>

#include <algorithm>
#include <random>
#include <vector>

#include "GS/GSRegs.h"

namespace
{
	GIFTag MakePackedTag(const std::vector<u8>& descs, u32 nloop)
	{
		GIFTag t = {};
		t.NLOOP = nloop;
		t.NREG = static_cast<u32>(descs.size()) & 0xF; // 16 registers encode as 0
		t.FLG = GIF_FLG_PACKED;
		u64 regs = 0;
		for (size_t i = 0; i < descs.size(); i++)
			regs |= static_cast<u64>(descs[i] & 0xF) << (i * 4);
		t.REGS = regs;
		return t;
	}

	GIFPath ClassifyTag(const std::vector<u8>& descs, u32 nloop = 7)
	{
		const GIFTag t = MakePackedTag(descs, nloop);
		GIFPath path = {};
		path.SetTag(&t);
		return path;
	}

	GSVector4i PackDescs(const std::vector<u8>& descs)
	{
		GSVector4i regs = GSVector4i::zero();
		for (size_t i = 0; i < descs.size(); i++)
			regs.U8[i] = descs[i] & 0xF;
		return regs;
	}

	GIFShapePlan Plan(const std::vector<u8>& descs)
	{
		GIFShapePlan plan = {};
		GIFBuildShapePlan(PackDescs(descs), static_cast<u32>(descs.size()), plan);
		return plan;
	}

	bool DescEscapes(u8 d)
	{
		return d == GIF_REG_PRIM || d == GIF_REG_TEX0_1 || d == GIF_REG_TEX0_2 ||
			   d == GIF_REG_CLAMP_1 || d == GIF_REG_CLAMP_2 || d == GIF_REG_A_D;
	}

	bool DescIsPosition(u8 d)
	{
		return d == GIF_REG_XYZF2 || d == GIF_REG_XYZ2 || d == GIF_REG_XYZF3 || d == GIF_REG_XYZ3;
	}

	// gtest prints a u8 as a character; every offset below is compared as an int.
	int Off(u8 v) { return static_cast<int>(v); }
	constexpr int kCarried = static_cast<int>(GIF_SHAPE_CARRIED);
} // namespace

// ---------------------------------------------------------------------------
// The descriptor alphabet.
// ---------------------------------------------------------------------------

TEST(GifShapePlan, EveryDescriptorValueIsClassified)
{
	// One tag per descriptor value, ahead of a position, so an accepted
	// descriptor shows up as a source of that vertex and a rejected one is the
	// only reason the tag could be rejected.
	for (u8 d = 0; d < 16; d++)
	{
		const GIFShapePlan p = Plan({d, GIF_REG_XYZ2});
		SCOPED_TRACE(testing::Message() << "descriptor 0x" << std::hex << static_cast<int>(d));

		if (DescEscapes(d))
		{
			EXPECT_EQ(p.status, GIFShapeStatus::Escape);
			continue;
		}

		ASSERT_EQ(p.status, GIFShapeStatus::Ok);
		EXPECT_EQ(Off(p.period), 2);

		if (DescIsPosition(d))
		{
			// Two positions, so two vertices, and the leading one carries
			// everything.
			ASSERT_EQ(Off(p.count), 2);
			EXPECT_EQ(Off(p.vertices[0].off), 0);
			EXPECT_EQ(Off(p.vertices[0].desc), static_cast<int>(d));
			EXPECT_EQ(p.vertices[0].IsXYZF(), d == GIF_REG_XYZF2 || d == GIF_REG_XYZF3);
			EXPECT_EQ(p.vertices[0].IsADC(), d == GIF_REG_XYZF3 || d == GIF_REG_XYZ3);
			EXPECT_EQ(Off(p.vertices[1].off), 1);
			// The trailing XYZ2 keeps the fog the leading position left, which is
			// its own qword only for the two that carry an F.
			EXPECT_EQ(Off(p.vertices[1].src.fog), p.vertices[0].IsXYZF() ? 0 : kCarried);
			continue;
		}

		ASSERT_EQ(Off(p.count), 1);
		const GIFShapeSources& s = p.vertices[0].src;
		EXPECT_EQ(Off(p.vertices[0].off), 1);
		EXPECT_EQ(Off(s.st), d == GIF_REG_STQ ? 0 : kCarried);
		EXPECT_EQ(Off(s.rgba), d == GIF_REG_RGBA ? 0 : kCarried);
		EXPECT_EQ(Off(s.uv), d == GIF_REG_UV ? 0 : kCarried);
		EXPECT_EQ(Off(s.fog), d == GIF_REG_FOG ? 0 : kCarried);
		EXPECT_FALSE(p.vertices[0].IsXYZF());
		// A colour write with no STQ before it copies the Q latch as it stood.
		EXPECT_EQ(Off(s.q), kCarried);
	}
}

TEST(GifShapePlan, TheTwoEmptyDescriptorsAreTransparent)
{
	// GIF_REG_INVALID is not in the packed handler table, so it keeps
	// GIFPackedRegHandlerNull and does exactly what a NOP does.
	const GIFShapePlan bare = Plan({GIF_REG_STQ, GIF_REG_RGBA, GIF_REG_XYZF2});
	const GIFShapePlan padded =
		Plan({GIF_REG_NOP, GIF_REG_STQ, GIF_REG_INVALID, GIF_REG_RGBA, GIF_REG_XYZF2});

	ASSERT_EQ(bare.status, GIFShapeStatus::Ok);
	ASSERT_EQ(padded.status, GIFShapeStatus::Ok);
	ASSERT_EQ(Off(bare.count), 1);
	ASSERT_EQ(Off(padded.count), 1);

	EXPECT_EQ(Off(bare.vertices[0].src.st), 0);
	EXPECT_EQ(Off(bare.vertices[0].src.rgba), 1);
	EXPECT_EQ(Off(bare.vertices[0].src.q), 0);
	EXPECT_EQ(Off(bare.vertices[0].off), 2);

	EXPECT_EQ(Off(padded.vertices[0].src.st), 1);
	EXPECT_EQ(Off(padded.vertices[0].src.rgba), 3);
	EXPECT_EQ(Off(padded.vertices[0].src.q), 1);
	EXPECT_EQ(Off(padded.vertices[0].off), 4);
}

// ---------------------------------------------------------------------------
// Agreement with GIFPath::SetTag.
// ---------------------------------------------------------------------------

TEST(GifShapePlan, MatchesTheLayoutTheClassifierWrote)
{
	constexpr int kSkip = -1;

	struct Case
	{
		const char* name;
		std::vector<u8> descs;
		u32 type;
		int off_xyz, st, q, rgba, uv;
		bool xyzf;
		// GIFPackedLayout as SetTag leaves it. The two contiguous triples never
		// read it -- their offsets are compile-time -- so it is not filled in.
		int stride, off_a, off_rgba;
	};

	const Case cases[] = {
		{"3:2,1,4", {2, 1, 4}, GIFPath::TYPE_STQRGBAXYZF2, 2, 0, 0, 1, kCarried, true, kSkip, kSkip, kSkip},
		{"3:2,1,5", {2, 1, 5}, GIFPath::TYPE_STQRGBAXYZ2, 2, 0, 0, 1, kCarried, false, kSkip, kSkip, kSkip},
		{"4:2,f,1,4", {2, 0xF, 1, 4}, GIFPath::TYPE_NOPSTQRGBAXYZF2, 3, 0, 0, 2, kCarried, true, 4, 0, 2},
		{"4:f,2,1,4", {0xF, 2, 1, 4}, GIFPath::TYPE_NOPSTQRGBAXYZF2, 3, 1, 1, 2, kCarried, true, 4, 1, 2},
		{"2:2,5", {2, 5}, GIFPath::TYPE_STQXYZ2, 1, 0, kCarried, kCarried, kCarried, false, 2, 0, kSkip},
		{"2:3,5", {3, 5}, GIFPath::TYPE_UVXYZ2, 1, kCarried, kCarried, kCarried, 0, false, 2, 0, kSkip},
		{"2:1,5", {1, 5}, GIFPath::TYPE_RGBAQXYZ2, 1, kCarried, kCarried, 0, kCarried, false, 2, 0, kSkip},
		{"1:4", {4}, GIFPath::TYPE_XYZF2ONLY, 0, kCarried, kCarried, kCarried, kCarried, true, 1, kSkip, kSkip},
		{"4:f,f,f,4", {0xF, 0xF, 0xF, 4}, GIFPath::TYPE_XYZF2ONLY, 3, kCarried, kCarried, kCarried, kCarried, true, 4, kSkip, kSkip},
		// SetTag halves nreg on a repeated pair, so the plan is built over the
		// period it left behind, not the one the tag spelled.
		{"4:2,5,2,5", {2, 5, 2, 5}, GIFPath::TYPE_STQXYZ2, 1, 0, kCarried, kCarried, kCarried, false, 2, 0, kSkip},
	};

	for (const Case& c : cases)
	{
		SCOPED_TRACE(c.name);
		const GIFPath path = ClassifyTag(c.descs);
		ASSERT_EQ(path.type, c.type) << "the classifier moved under the plan";

		GIFShapePlan p = {};
		ASSERT_TRUE(GIFBuildShapePlan(path.regs, path.nreg, p));
		ASSERT_EQ(Off(p.count), 1) << "every layout the classifier names is one vertex a period";
		EXPECT_EQ(Off(p.period), static_cast<int>(path.nreg));

		const GIFShapeVertex& v = p.vertices[0];
		EXPECT_EQ(Off(v.off), c.off_xyz);
		EXPECT_EQ(Off(v.src.st), c.st);
		EXPECT_EQ(Off(v.src.q), c.q);
		EXPECT_EQ(Off(v.src.rgba), c.rgba);
		EXPECT_EQ(Off(v.src.uv), c.uv);
		// An XYZF2 vertex reads fog out of its own position qword.
		EXPECT_EQ(Off(v.src.fog), c.xyzf ? c.off_xyz : kCarried);
		EXPECT_EQ(v.IsXYZF(), c.xyzf);
		EXPECT_FALSE(v.IsADC());

		if (c.stride != kSkip)
		{
			EXPECT_EQ(static_cast<int>(path.layout.stride), c.stride);
			EXPECT_EQ(static_cast<int>(path.layout.off_xyz), c.off_xyz);
		}
		if (c.off_a != kSkip)
			EXPECT_EQ(static_cast<int>(path.layout.off_a), c.off_a);
		if (c.off_rgba != kSkip)
			EXPECT_EQ(static_cast<int>(path.layout.off_rgba), c.off_rgba);
	}
}

// ---------------------------------------------------------------------------
// The shapes a census of the Sly 3 dump found on the per-qword path.
// ---------------------------------------------------------------------------

TEST(GifShapePlan, CensusShapes)
{
	// The two biggest, and both are out of reach: an A+D writes a GS register.
	EXPECT_EQ(Plan({0xE, 2, 1, 4, 2, 4, 2, 4, 2, 4}).status, GIFShapeStatus::Escape);
	EXPECT_EQ(Plan({0xE, 4, 4}).status, GIFShapeStatus::Escape);
	// Three A+Ds, an STQ and a colour, and no position anywhere: rejected for
	// the register writes, which are looked at before the missing vertex.
	EXPECT_EQ(Plan({0xE, 0xE, 0xE, 2, 1}).status, GIFShapeStatus::Escape);

	// {RGBAQ, XYZF2}: the XYZF2 twin the classifier does not take.
	const GIFShapePlan rgba = Plan({1, 4});
	ASSERT_EQ(rgba.status, GIFShapeStatus::Ok);
	ASSERT_EQ(Off(rgba.count), 1);
	EXPECT_EQ(Off(rgba.vertices[0].src.rgba), 0);
	EXPECT_EQ(Off(rgba.vertices[0].src.q), kCarried);
	EXPECT_EQ(Off(rgba.vertices[0].src.st), kCarried);
	EXPECT_EQ(Off(rgba.vertices[0].off), 1);
	EXPECT_TRUE(rgba.vertices[0].IsXYZF());

	// {UV, XYZF2}, off the sprite stream.
	const GIFShapePlan uv = Plan({3, 4});
	ASSERT_EQ(uv.status, GIFShapeStatus::Ok);
	EXPECT_EQ(Off(uv.vertices[0].src.uv), 0);
	EXPECT_TRUE(uv.vertices[0].IsXYZF());

	// A bare position, and the NOP-padded spelling of the same thing.
	EXPECT_EQ(Plan({5}).status, GIFShapeStatus::Ok);
	EXPECT_FALSE(Plan({5}).vertices[0].IsXYZF());
	EXPECT_EQ(Plan({0xF, 0xF, 0xF, 4}).status, GIFShapeStatus::Ok);

	// The triple with a NOP anywhere in it, both spellings the fan stream sends.
	EXPECT_EQ(Plan({2, 1, 0xF, 4}).status, GIFShapeStatus::Ok);
	EXPECT_EQ(Plan({2, 0xF, 1, 4}).status, GIFShapeStatus::Ok);
}

TEST(GifShapePlan, ManyVerticesToAPeriod)
{
	// The census's biggest shape without its leading A+D: one colour for the
	// strip, then a position per vertex with its own ST. GIFPackedLayout cannot
	// describe this -- it names one vertex per stride.
	const GIFShapePlan p = Plan({2, 1, 4, 2, 4, 2, 4, 2, 4});
	ASSERT_EQ(p.status, GIFShapeStatus::Ok);
	ASSERT_EQ(Off(p.count), 4);
	EXPECT_EQ(Off(p.period), 9);

	const int st[4] = {0, 3, 5, 7};
	const int xyz[4] = {2, 4, 6, 8};
	for (int i = 0; i < 4; i++)
	{
		SCOPED_TRACE(i);
		EXPECT_EQ(Off(p.vertices[i].src.st), st[i]);
		EXPECT_EQ(Off(p.vertices[i].off), xyz[i]);
		// The colour, and the Q it copied, are the same qword for all four.
		EXPECT_EQ(Off(p.vertices[i].src.rgba), 1);
		EXPECT_EQ(Off(p.vertices[i].src.q), 0);
	}

	EXPECT_EQ(Off(p.tail.st), 7);
	EXPECT_EQ(Off(p.tail.rgba), 1);
}

// ---------------------------------------------------------------------------
// The rejections.
// ---------------------------------------------------------------------------

TEST(GifShapePlan, PeriodsThatBuildNothing)
{
	EXPECT_EQ(Plan({0xF}).status, GIFShapeStatus::NoVertex);
	EXPECT_EQ(Plan({0xF, 0xF, 0xF, 0xF}).status, GIFShapeStatus::NoVertex);
	EXPECT_EQ(Plan({0xB}).status, GIFShapeStatus::NoVertex);
	EXPECT_EQ(Plan({2, 1}).status, GIFShapeStatus::NoVertex);
}

TEST(GifShapePlan, PeriodsThatDoNotRepeat)
{
	// A source written after the only position: the next period's vertex reads
	// what this one wrote, and the period does not describe itself.
	EXPECT_EQ(Plan({4, 2}).status, GIFShapeStatus::CrossPeriod);
	EXPECT_EQ(Plan({4, 1}).status, GIFShapeStatus::CrossPeriod);
	EXPECT_EQ(Plan({4, 3}).status, GIFShapeStatus::CrossPeriod);
	EXPECT_EQ(Plan({5, 0xA}).status, GIFShapeStatus::CrossPeriod);

	// An XYZF2 writes the fog register from its own qword, so an XYZ2 beside it
	// reads that and not what the tag last spelled. Which way round they sit
	// decides whether the period describes itself.
	EXPECT_EQ(Plan({4, 0xA}).status, GIFShapeStatus::Ok) << "no vertex reads the trailing fog write";
	EXPECT_EQ(Off(Plan({4, 5}).vertices[1].src.fog), 0);
	EXPECT_EQ(Plan({5, 4}).status, GIFShapeStatus::CrossPeriod);

	// The Q lag on its own. Both writes precede the position and both are read
	// by this vertex, but the colour at qword 0 copies a Q the STQ at qword 1
	// only sets afterwards -- so the vertex's Q comes from the period before.
	EXPECT_EQ(Plan({1, 2, 4}).status, GIFShapeStatus::CrossPeriod);
	EXPECT_EQ(Plan({2, 1, 4}).status, GIFShapeStatus::Ok);

	// A trailing write the next period overwrites before its own vertex reads
	// it is not a cross-period read. The period still latches it, which is what
	// the tail is for.
	const GIFShapePlan p = Plan({2, 1, 4, 2});
	ASSERT_EQ(p.status, GIFShapeStatus::Ok);
	EXPECT_EQ(Off(p.vertices[0].src.st), 0);
	EXPECT_EQ(Off(p.tail.st), 3);
}

// ---------------------------------------------------------------------------
// Against a flat walk that has no notion of a period.
// ---------------------------------------------------------------------------

namespace
{
	constexpr int kNone = -1;

	struct FlatSources
	{
		int st, q, rgba, uv, fog;
	};

	struct FlatVertex
	{
		FlatSources src;
		int off;
		u8 desc;
	};

	std::vector<FlatVertex> FlatWalk(const std::vector<u8>& descs, u32 periods)
	{
		std::vector<FlatVertex> out;
		FlatSources s = {kNone, kNone, kNone, kNone, kNone};

		for (u32 p = 0; p < periods; p++)
		{
			for (size_t i = 0; i < descs.size(); i++)
			{
				const int at = static_cast<int>(p * descs.size() + i);
				switch (descs[i])
				{
					case GIF_REG_STQ: s.st = at; break;
					case GIF_REG_RGBA: s.rgba = at; s.q = s.st; break;
					case GIF_REG_UV: s.uv = at; break;
					case GIF_REG_FOG: s.fog = at; break;
					case GIF_REG_XYZF2:
					case GIF_REG_XYZF3:
						s.fog = at;
						[[fallthrough]];
					case GIF_REG_XYZ2:
					case GIF_REG_XYZ3:
						out.push_back({s, at, descs[i]});
						break;
					default: break;
				}
			}
		}

		return out;
	}

	// What the plan says vertex `j` of period `p` reads, in flat-walk terms.
	int Expand(u8 src, u32 period, u32 nreg)
	{
		return src == GIF_SHAPE_CARRIED ? kNone : static_cast<int>(period * nreg + src);
	}

	void CheckAgainstFlatWalk(const std::vector<u8>& descs)
	{
		constexpr u32 kPeriods = 4;
		const u32 nreg = static_cast<u32>(descs.size());
		const GIFShapePlan p = Plan(descs);
		const std::vector<FlatVertex> flat = FlatWalk(descs, kPeriods);

		const bool escapes = std::any_of(descs.begin(), descs.end(), DescEscapes);
		const size_t per_period =
			static_cast<size_t>(std::count_if(descs.begin(), descs.end(), DescIsPosition));

		if (escapes)
		{
			ASSERT_EQ(p.status, GIFShapeStatus::Escape);
			return;
		}
		ASSERT_NE(p.status, GIFShapeStatus::Escape);

		if (per_period == 0)
		{
			ASSERT_EQ(p.status, GIFShapeStatus::NoVertex);
			return;
		}
		ASSERT_NE(p.status, GIFShapeStatus::NoVertex);

		// Period 1's vertices decide it: a field of one of them that the flat
		// walk sources from period 0 is a read the plan cannot express.
		bool crossed = false;
		for (size_t j = 0; j < per_period; j++)
		{
			const FlatSources& s = flat[per_period + j].src;
			for (int f : {s.st, s.q, s.rgba, s.uv, s.fog})
				crossed |= (f != kNone && f < static_cast<int>(nreg));
		}
		ASSERT_EQ(crossed, p.status == GIFShapeStatus::CrossPeriod);

		if (crossed)
			return;

		ASSERT_EQ(p.status, GIFShapeStatus::Ok);
		ASSERT_EQ(static_cast<size_t>(p.count), per_period);
		ASSERT_EQ(Off(p.period), static_cast<int>(nreg));

		for (u32 period = 0; period < kPeriods; period++)
		{
			for (size_t j = 0; j < per_period; j++)
			{
				SCOPED_TRACE(testing::Message() << "period " << period << " vertex " << j);
				const FlatVertex& f = flat[period * per_period + j];
				const GIFShapeVertex& v = p.vertices[j];

				EXPECT_EQ(f.off, Expand(v.off, period, nreg));
				EXPECT_EQ(static_cast<int>(f.desc), Off(v.desc));
				EXPECT_EQ(f.src.st, Expand(v.src.st, period, nreg));
				EXPECT_EQ(f.src.q, Expand(v.src.q, period, nreg));
				EXPECT_EQ(f.src.rgba, Expand(v.src.rgba, period, nreg));
				EXPECT_EQ(f.src.uv, Expand(v.src.uv, period, nreg));
				EXPECT_EQ(f.src.fog, Expand(v.src.fog, period, nreg));
			}
		}
	}
} // namespace

TEST(GifShapePlan, RandomShapesAgreeWithAFlatWalk)
{
	// Three alphabets: everything, so escapes and rejections are the common
	// case; the descriptors that can produce a plan, so accepted shapes are; and
	// one heavy on NOPs, which is how the padded layouts are actually spelled.
	const std::vector<u8> all = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0xA, 0xB, 0xC, 0xD, 0xE, 0xF};
	const std::vector<u8> planar = {1, 2, 3, 4, 5, 0xA, 0xB, 0xC, 0xD, 0xF};
	const std::vector<u8> padded = {0xF, 0xF, 0xF, 0xF, 1, 2, 3, 4, 5, 0xA};

	std::mt19937 rng(0x5A1Fu);
	for (const std::vector<u8>* alphabet : {&all, &planar, &padded})
	{
		for (u32 nreg = 1; nreg <= 16; nreg++)
		{
			for (int iter = 0; iter < 400; iter++)
			{
				std::vector<u8> descs(nreg);
				for (u8& d : descs)
					d = (*alphabet)[rng() % alphabet->size()];

				SCOPED_TRACE(testing::Message() << "nreg " << nreg << " iter " << iter);
				CheckAgainstFlatWalk(descs);
				if (testing::Test::HasFatalFailure() || testing::Test::HasNonfatalFailure())
					return;
			}
		}
	}
}

TEST(GifShapePlan, EveryShortShapeAgreesWithAFlatWalk)
{
	// Exhaustive over one, two and three descriptors: 16 + 256 + 4096 tags, so
	// every ordering of every pair and triple of descriptor values is walked.
	for (u32 a = 0; a < 16; a++)
	{
		CheckAgainstFlatWalk({static_cast<u8>(a)});
		for (u32 b = 0; b < 16; b++)
		{
			CheckAgainstFlatWalk({static_cast<u8>(a), static_cast<u8>(b)});
			for (u32 c = 0; c < 16; c++)
				CheckAgainstFlatWalk({static_cast<u8>(a), static_cast<u8>(b), static_cast<u8>(c)});
		}
		if (testing::Test::HasFatalFailure() || testing::Test::HasNonfatalFailure())
			return;
	}
}
