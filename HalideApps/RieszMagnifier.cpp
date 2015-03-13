#include "stdafx.h"
#include "RieszMagnifier.h"
#include "Util.h"

using namespace Halide;

using std::vector;
using std::cerr;
using std::endl;

#define EPSILON 1e-3f

static Var x("x");
static Var y("y");
static Var xi("xi");
static Var yi("yi");
static Var c("c");
static Var p("p");

RieszMagnifier::RieszMagnifier(int channels, Halide::Type type, int pyramidLevels) :
    bandSigma(vector<float>(pyramidLevels)),
    channels(channels),
    pyramidLevels(pyramidLevels),
    a1(Param<float>("a1")),
    a2(Param<float>("a2")),
    b0(Param<float>("b0")),
    b1(Param<float>("b1")),
    b2(Param<float>("b2")),
    alpha(Param<float>("alpha")),
    pParam(Param<int>("pParam")),
    stabilize(Param<int>("stabilize")),
    output(Func("output"))
{
	if (channels!=1 && channels!=3) {
		throw std::invalid_argument("Channels must be either 1 or 3.");
    }

    if (type!=type_of<unsigned char>() && type!=type_of<float>()) {
		throw std::invalid_argument("Only 8-bit unsigned integer and 32-bit floating point types are supported.");
    }

	// Initialize pyramid buffer params
	for (int j=0; j<pyramidLevels; j++) {
		historyBuffer.push_back(ImageParam(type_of<float>(), 4, "historyBuffer" + std::to_string(j)));
		amplitudeBuffer.push_back(ImageParam(type_of<float>(), 2, "amplitudeBuffer" + std::to_string(j)));
    }

	// Initialize spatial regularization sigmas
	computeBandSigmas();

    // Input frame conversion to float
	input = ImageParam(type, channels == 3 ? 3 : 2, "input");
	Func floatInput("floatInput");
	if (type == UInt(8)) {
		floatInput(_) = cast<float>(input(_)) / 255.0f;
    } else {
		floatInput(_) = input(_);
    }

    // RGB to YCbCr conversion
	Func grey("grey"), cb("cb"), cr("cr");
	if (channels == 3) {
		grey(x,y) = 0.299f * floatInput(x,y, 0) + 0.587f * floatInput(x,y, 1) + 0.114f * floatInput(x,y, 2);
		cb(x,y) = -0.168736f * floatInput(x,y, 0) - 0.331264f * floatInput(x,y, 1) + 0.5f * floatInput(x,y, 2);
		cr(x,y) = 0.5f * floatInput(x,y, 0) - 0.418688f * floatInput(x,y, 1) - 0.081312f * floatInput(x,y, 2);
	}
	else {
		grey(x,y) = floatInput(x,y);
	}

	// Gaussian pyramid: implemented as separable X and Y Gaussian blurs
    // base level set to input image
	gPyramidDownX   = makeFuncArray(pyramidLevels, "gPyramidDownX");
	gPyramid        = makeFuncArray(pyramidLevels, "gPyramid");

    gPyramid[0](x,y) = grey(x,y);
	for (int j=1; j<pyramidLevels; j++) {
		gPyramidDownX[j](x,y) = downsampleG5X(clipToEdges(gPyramid[j - 1], scaleSize(input.width(), j - 1), scaleSize(input.height(), j - 1)))(x,y);
		gPyramid[j](x,y) = downsampleG5Y(gPyramidDownX[j])(x,y);
	}

	// Laplacian pyramid: implemented as difference of Gaussians
	lPyramidUpX = makeFuncArray(pyramidLevels, "lPyramidUpX");
	lPyramid      = makeFuncArray(pyramidLevels, "lPyramid");
	lPyramid_orig = makeFuncArray(pyramidLevels, "lPyramid_orig");

	lPyramid_orig[pyramidLevels-1](x,y) = gPyramid[pyramidLevels-1](x,y);
	//lPyramid     [pyramidLevels-1](x,y) = gPyramid[pyramidLevels-1](x,y);

    for (int j=pyramidLevels-2; j>=0; j--) {
		lPyramidUpX[j](x,y) = upsampleG5X(clipToEdges(gPyramid[j + 1], scaleSize(input.width(), j + 1), scaleSize(input.height(), j + 1)))(x,y);
		lPyramid_orig[j](x,y) = gPyramid[j](x,y) - upsampleG5Y(lPyramidUpX[j])(x,y);
	}

	// Amplitude, R1 and R2 pyramid
    amp         = makeFuncArray(pyramidLevels, "amp");         // Amplitude of current frame
    ampPrev     = makeFuncArray(pyramidLevels, "ampPrev");     // Amplitude of prev frame
    amp_orig    = makeFuncArray(pyramidLevels, "amp_orig");    // Amplitude of current frame without any stabilization
	r1Pyramid   = makeFuncArray(pyramidLevels, "r1Pyramid");   // R1 pyramid
	r2Pyramid   = makeFuncArray(pyramidLevels, "r2Pyramid");   // R2 pyramid
	r1Prev      = makeFuncArray(pyramidLevels, "r1Prev");      // R1 pyramid of prev frame
	r2Prev      = makeFuncArray(pyramidLevels, "r2Prev");      // R2 pyramid of prev frame

    // computation of R1 and R2 pyramids
    for (int j=0; j<pyramidLevels; j++) {
        Func clampedLPyramid, clampedLPyramidPrev;
        Func r1Pyramid_orig, r2Pyramid_orig;

        float coeff = 0.6f;

        clampedLPyramid(x,y) = clipToEdges(lPyramid_orig[j],scaleSize(input.width(),j), scaleSize(input.height(),j))(x,y);
        r1Pyramid_orig (x,y) = coeff*(clampedLPyramid(x+1,y) - clampedLPyramid(x-1,y));
        r2Pyramid_orig (x,y) = coeff*(clampedLPyramid(x,y+1) - clampedLPyramid(x,y-1));

        amp_orig [j](x,y) = hypot(clampedLPyramid(x,y), hypot(r1Pyramid_orig(x,y), r2Pyramid_orig(x,y))) + EPSILON;
        amp      [j](x,y) = select(stabilize==0, amp_orig[j](x,y), amplitudeBuffer[j](x,y));
        r1Pyramid[j](x,y) = r1Pyramid_orig(x,y)  * select(stabilize==0, 1.0f, amp[j](x,y)/amp_orig[j](x,y));
        r2Pyramid[j](x,y) = r2Pyramid_orig(x,y)  * select(stabilize==0, 1.0f, amp[j](x,y)/amp_orig[j](x,y));
        lPyramid [j](x,y) = lPyramid_orig[j](x,y)* select(stabilize==0, 1.0f, amp[j](x,y)/amp_orig[j](x,y));

        clampedLPyramidPrev(x,y,p) = clipToEdges(historyBuffer[j])(x,y,0,p);
        r1Prev [j](x,y) = coeff*(clampedLPyramidPrev(x+1,y,(pParam+1)%2) - clampedLPyramidPrev(x-1,y,(pParam+1)%2));
        r2Prev [j](x,y) = coeff*(clampedLPyramidPrev(x,y+1,(pParam+1)%2) - clampedLPyramidPrev(x,y-1,(pParam+1)%2));
        ampPrev[j](x,y) = hypot(clampedLPyramidPrev(x,y,(pParam+1)%2), hypot(r1Prev[j](x,y), r2Prev[j](x,y)));
    }
    lPyramidCopy = copyPyramidToCircularBuffer(pyramidLevels, lPyramid, historyBuffer, 0, pParam, "lPyramidCopy");

    // quaternionic phase difference as a tuple
    phi_diff    = makeFuncArray(pyramidLevels, "phi_diff");
    qPhaseDiffC = makeFuncArray(pyramidLevels, "qPhaseDiffC");
    qPhaseDiffS = makeFuncArray(pyramidLevels, "qPhaseDiffS");

    // computation of r x r_prev^-1 = r x r_prev_conjugate
    // cos(phi) = (r x r_prev^-1) / ||r x r_prev||
    // phi = phi_m - phi_prev
    // qPhaseDiffC = phi - phi_prev * cos(theta);
    // qPhaseDiffS = phi - phi_prev * sin(theta);
    qPhaseDiffC = makeFuncArray(pyramidLevels, "qPhaseDiffC");
    qPhaseDiffS = makeFuncArray(pyramidLevels, "qPhaseDiffS");

    for (int j=0; j<pyramidLevels; j++) {
        Func productReal, productI, productJ, ijAmplitude, amplitude;

        Expr a = 1.0f; //ampPrev[j](x,y);
        Expr A = 1.0f; //amp[j](x,y);

        Expr I  = select(A<3*EPSILON, 0.0f, lPyramidCopy[j](x,y)/A);
        Expr R1 = select(A<3*EPSILON, 0.0f, r1Pyramid[j](x,y)   /A);
        Expr R2 = select(A<3*EPSILON, 0.0f, r2Pyramid[j](x,y)   /A);

        Expr i  = select(a<3*EPSILON, 0.0f, historyBuffer[j](x,y,0,(pParam+1)%2)/a);
        Expr r1 = select(a<3*EPSILON, 0.0f, r1Prev[j](x,y)/a);
        Expr r2 = select(a<3*EPSILON, 0.0f, r2Prev[j](x,y)/a);

        productReal(x,y) = I*i + R1*r1 + R2*r2;
        productI(x,y)    = R1*i - r1*I;
        productJ(x,y)    = R2*i - r2*I;

        ijAmplitude(x,y) = hypot(productI(x,y),    productJ(x,y))    + EPSILON;
        amplitude  (x,y) = hypot(ijAmplitude(x,y), productReal(x,y)) + EPSILON;

        phi_diff[j](x,y)    = acos(productReal(x,y) / amplitude(x,y)) / ijAmplitude(x,y);

        qPhaseDiffC[j](x,y) = productI(x,y) * phi_diff[j](x,y);
        qPhaseDiffS[j](x,y) = productJ(x,y) * phi_diff[j](x,y);
    }

    // Cumulative sums on phi to give
    // qPhaseC = cumsum(phi - phi_prev) * cos(theta);
    // qPhaseS = cumsum(phi - phi_prev) * sin(theta);
    phaseC = makeFuncArray(pyramidLevels, "phaseC");
    phaseS = makeFuncArray(pyramidLevels, "phaseS");
    for (int j=0; j<pyramidLevels; j++) {
        phaseC[j](x,y) = qPhaseDiffC[j](x,y) + historyBuffer[j](x,y, 1, (pParam + 1) % 2);
        phaseS[j](x,y) = qPhaseDiffS[j](x,y) + historyBuffer[j](x,y, 2, (pParam + 1) % 2);
    }

    phaseCCopy  = copyPyramidToCircularBuffer(pyramidLevels, phaseC, historyBuffer, 1, pParam, "phaseCCopy");
    phaseSCopy  = copyPyramidToCircularBuffer(pyramidLevels, phaseS, historyBuffer, 2, pParam, "phaseSCopy");

    changeC     = makeFuncArray(pyramidLevels, "changeC");
    lowpass1C   = makeFuncArray(pyramidLevels, "lowpass1C");
    lowpass2C   = makeFuncArray(pyramidLevels, "lowpass2C");
    changeS     = makeFuncArray(pyramidLevels, "changeS");
    lowpass1S   = makeFuncArray(pyramidLevels, "lowpass1S");
    lowpass2S   = makeFuncArray(pyramidLevels, "lowpass2S");

    // temporal filtering of unwrapped phase
    // Linear filter. Order of evaluation here is important.
    for (int j=0; j<pyramidLevels; j++) {
        changeC[j](x,y)   = b0 * phaseCCopy[j](x,y) + historyBuffer[j](x,y, 3, (pParam + 1) % 2);
        lowpass1C[j](x,y) = b1 * phaseCCopy[j](x,y) + historyBuffer[j](x,y, 4, (pParam + 1) % 2) - a1 * changeC[j](x,y);
        lowpass2C[j](x,y) = b2 * phaseCCopy[j](x,y) - a2 * changeC[j](x,y);

        changeS[j](x,y)   = b0 * phaseSCopy[j](x,y) + historyBuffer[j](x,y, 5, (pParam + 1) % 2);
        lowpass1S[j](x,y) = b1 * phaseSCopy[j](x,y) + historyBuffer[j](x,y, 6, (pParam + 1) % 2) - a1 * changeS[j](x,y);
        lowpass2S[j](x,y) = b2 * phaseSCopy[j](x,y) - a2 * changeS[j](x,y);
    }

    lowpass1CCopy = copyPyramidToCircularBuffer(pyramidLevels, lowpass1C, historyBuffer, 3, pParam, "lowpass1CCopy");
    lowpass2CCopy = copyPyramidToCircularBuffer(pyramidLevels, lowpass2C, historyBuffer, 4, pParam, "lowpass2CCopy");
    lowpass1SCopy = copyPyramidToCircularBuffer(pyramidLevels, lowpass1S, historyBuffer, 5, pParam, "lowpass1SCopy");
    lowpass2SCopy = copyPyramidToCircularBuffer(pyramidLevels, lowpass2S, historyBuffer, 6, pParam, "lowpass2SCopy");

    changeCTuple  = makeFuncArray(pyramidLevels, "changeCTuple");
    changeSTuple  = makeFuncArray(pyramidLevels, "changeSTuple");
    changeC2      = makeFuncArray(pyramidLevels, "changeC2");
    changeS2      = makeFuncArray(pyramidLevels, "changeS2");

    changeCAmp    = makeFuncArray(pyramidLevels, "changeCAmp");
    changeCRegX   = makeFuncArray(pyramidLevels, "changeCRegX");
    changeCReg    = makeFuncArray(pyramidLevels, "changeCReg");
    changeSAmp    = makeFuncArray(pyramidLevels, "changeSAmp");
    changeSRegX   = makeFuncArray(pyramidLevels, "changeSRegX");
    changeSReg    = makeFuncArray(pyramidLevels, "changeSReg");
    ampRegX       = makeFuncArray(pyramidLevels, "ampRegX");
    ampReg        = makeFuncArray(pyramidLevels, "ampReg");
    magC          = makeFuncArray(pyramidLevels, "magC");
    pair          = makeFuncArray(pyramidLevels, "pair");
    outLPyramid   = makeFuncArray(pyramidLevels, "outLPyramid");

    for (int j=0; j<pyramidLevels; j++) {
        //
        changeCTuple[j](x,y) = { changeC[j](x,y), lowpass1CCopy[j](x,y), lowpass2CCopy[j](x,y) };
		changeSTuple[j](x,y) = { changeS[j](x,y), lowpass1SCopy[j](x,y), lowpass2SCopy[j](x,y) };
		changeC2[j](x,y) = changeCTuple[j](x,y)[0];
		changeS2[j](x,y) = changeSTuple[j](x,y)[0];

        // spatial Gaussian blur
		float sigma = bandSigma[j];

		ampRegX[j](x,y) = gaussianBlurX(clipToEdges(amp[j], scaleSize(input.width(), j), scaleSize(input.height(), j)), sigma)(x,y);
		ampReg[j](x,y) = gaussianBlurY(ampRegX[j], sigma)(x,y);

		changeCAmp[j](x,y) = changeC2[j](x,y) * amp[j](x,y);
		changeCRegX[j](x,y) = gaussianBlurX(clipToEdges(changeCAmp[j], scaleSize(input.width(), j), scaleSize(input.height(), j)), sigma)(x,y);
		changeCReg[j](x,y) = gaussianBlurY(changeCRegX[j], sigma)(x,y) / ampReg[j](x,y);

		changeSAmp[j](x,y) = changeS2[j](x,y) * amp[j](x,y);
		changeSRegX[j](x,y) = gaussianBlurX(clipToEdges(changeSAmp[j], scaleSize(input.width(), j), scaleSize(input.height(), j)), sigma)(x,y);
		changeSReg[j](x,y) = gaussianBlurY(changeSRegX[j], sigma)(x,y) / ampReg[j](x,y);

		Expr creg = (sigma == 0.0f ? changeC2[j](x,y) : changeCReg[j](x,y));
		Expr sreg = (sigma == 0.0f ? changeS2[j](x,y) : changeSReg[j](x,y));
		magC[j](x,y) = hypot(creg, sreg) + EPSILON;

        // magnifying
		pair[j](x,y) = (r1Pyramid[j](x,y) * creg + r2Pyramid[j](x,y) * sreg) / magC[j](x,y);
		outLPyramid[j](x,y) = lPyramid[j](x,y)*cos(alpha*magC[j](x,y)) - pair[j](x,y)*sin(alpha*magC[j](x,y));
	}

	outGPyramidUpX = makeFuncArray(pyramidLevels, "outGPyramidUpX");
	outGPyramid    = makeFuncArray(pyramidLevels+1, "outGPyramid");
	outGPyramid[pyramidLevels-1](x,y) = lPyramid[pyramidLevels-1](x,y);

    for (int j=pyramidLevels-2; j>=0; j--) {
		outGPyramidUpX[j](x,y) = upsampleG5X(clipToEdges(outGPyramid[j + 1], scaleSize(input.width(), j + 1), scaleSize(input.height(), j + 1)))(x,y);
		outGPyramid[j](x,y) = outLPyramid[j](x,y) + upsampleG5Y(outGPyramidUpX[j])(x,y);
	}

#if 0
    // YCrCb -> RGB
    floatOutput(x,y, c) = clamp(select(
    	c == 0, outGPyramid[0](x,y) + 1.402f * cr(x,y),
    	c == 1, outGPyramid[0](x,y) - 0.34414f * cb(x,y) - 0.71414f * cr(x,y),
    	outGPyramid[0](x,y) + 1.772f * cb(x,y)), 0.0f, 1.0f);

    output(x,y,c) = (type==type_of<unsigned char>()
            ? cast<unsigned char>(floatOutput(x,y,c)*255.0f)
            : floatOutput(x,y, c));
#else
    output(x,y,c) = phi_diff[0](x,y);
#endif
}

void RieszMagnifier::schedule(bool tile)
{
#if 1
	int VECTOR_SIZE = 8;
	int THREAD_SIZE = 16;

    output.parallel(y, THREAD_SIZE).vectorize(x, VECTOR_SIZE).reorder(c,x,y).bound(c, 0, channels).unroll(c);

	for (int j=0; j<pyramidLevels; j++) {
        outGPyramid[j]   .compute_root();
        outGPyramidUpX[j].compute_root();

        ampReg[j]       .compute_root().split(y, y, yi, 16);
        ampRegX[j]      .compute_at(ampReg[j], y);
        changeCReg[j]   .compute_root().split(y, y, yi, 16);
        changeCRegX[j]  .compute_at(changeCReg[j], y);
        changeSReg[j]   .compute_root().split(y, y, yi, 16);
        changeSRegX[j]  .compute_at(changeSReg[j], y);
        amp[j]          .compute_root();
        ampPrev[j]      .compute_root();
        amp_orig[j]     .compute_root();

        changeCTuple[j] .compute_root();
        changeSTuple[j] .compute_root();

        lowpass1CCopy[j].compute_root();
        lowpass2CCopy[j].compute_root();
        lowpass1SCopy[j].compute_root();
        lowpass2SCopy[j].compute_root();
        lowpass1C[j]    .compute_root();
        lowpass2C[j]    .compute_root();
        lowpass1S[j]    .compute_root();
        lowpass2S[j]    .compute_root();

        phaseCCopy[j]   .compute_root();
        phaseSCopy[j]   .compute_root();
        phaseC[j]       .compute_root();
        phaseS[j]       .compute_root();

        r1Pyramid[j]    .compute_root();
        r2Pyramid[j]    .compute_root();

        lPyramidCopy[j] .compute_root();
        lPyramid_orig[j].compute_root().split(y, y, yi, 8);
        lPyramid[j]     .compute_root().split(y, y, yi, 8);
        lPyramidUpX[j]  .compute_at(lPyramid_orig[j], y);

		if (j>0) {
			gPyramid[j]     .compute_root().split(y, y, yi, 8);
			gPyramidDownX[j].compute_at(gPyramid[j], y);
            gPyramid[j]     .parallel(y, THREAD_SIZE).vectorize(x, VECTOR_SIZE);
            gPyramidDownX[j].parallel(y, THREAD_SIZE).vectorize(x, VECTOR_SIZE);
		}

		if (j<=4) {
			outGPyramid[j]   .parallel(y,THREAD_SIZE).vectorize(x, VECTOR_SIZE);
			outGPyramidUpX[j].parallel(y,THREAD_SIZE).vectorize(x, VECTOR_SIZE);

			ampReg[j]       .parallel(y, THREAD_SIZE).vectorize(x, VECTOR_SIZE);
			ampRegX[j]      .parallel(y, THREAD_SIZE).vectorize(x, VECTOR_SIZE);
			changeCReg[j]   .parallel(y, THREAD_SIZE).vectorize(x, VECTOR_SIZE);
			changeCRegX[j]  .parallel(y, THREAD_SIZE).vectorize(x, VECTOR_SIZE);
			changeSReg[j]   .parallel(y, THREAD_SIZE).vectorize(x, VECTOR_SIZE);
			changeSRegX[j]  .parallel(y, THREAD_SIZE).vectorize(x, VECTOR_SIZE);
			amp[j]          .parallel(y, THREAD_SIZE).vectorize(x, VECTOR_SIZE);
			ampPrev[j]      .parallel(y, THREAD_SIZE).vectorize(x, VECTOR_SIZE);
			amp_orig[j]     .parallel(y, THREAD_SIZE).vectorize(x, VECTOR_SIZE);

			changeCTuple[j] .parallel(y, THREAD_SIZE).vectorize(x, VECTOR_SIZE);
			changeSTuple[j] .parallel(y, THREAD_SIZE).vectorize(x, VECTOR_SIZE);

			lowpass1C[j]    .parallel(y, THREAD_SIZE).vectorize(x, VECTOR_SIZE);
			lowpass2C[j]    .parallel(y, THREAD_SIZE).vectorize(x, VECTOR_SIZE);
			lowpass1S[j]    .parallel(y, THREAD_SIZE).vectorize(x, VECTOR_SIZE);
			lowpass2S[j]    .parallel(y, THREAD_SIZE).vectorize(x, VECTOR_SIZE);

			phaseC[j]       .parallel(y, THREAD_SIZE).vectorize(x, VECTOR_SIZE);
			phaseS[j]       .parallel(y, THREAD_SIZE).vectorize(x, VECTOR_SIZE);

			r1Pyramid[j]    .parallel(y, THREAD_SIZE).vectorize(x, VECTOR_SIZE);
			r2Pyramid[j]    .parallel(y, THREAD_SIZE).vectorize(x, VECTOR_SIZE);

			lPyramid[j]     .parallel(y, THREAD_SIZE).vectorize(x, VECTOR_SIZE);
			lPyramidUpX[j]  .parallel(y, THREAD_SIZE).vectorize(x, VECTOR_SIZE);
		}
	}
#else
	const int VECTOR_SIZE = 8;

    if (channels == 3) {
		output.reorder(c, x,y).bound(c, 0, channels).unroll(c);
    }
	output.parallel(y, 4).vectorize(x, 4);

	if (tile) {
		output.tile(x,y, xi, yi, 80, 20);
	}

	for (int j=0; j<pyramidLevels; j++) {
		if (tile && j<=0) {
			outGPyramid[j].compute_at(output, x);
			outGPyramidUpX[j].compute_at(output, x);

			ampReg[j].compute_at(output, x);
			ampRegX[j].compute_at(output, x);
			changeCReg[j].compute_at(output, x);
			changeCRegX[j].compute_at(output, x);
			changeSReg[j].compute_at(output, x);
			changeSRegX[j].compute_at(output, x);
			amp[j].compute_at(output, x);

			changeCTuple[j].compute_at(output, x);
			changeSTuple[j].compute_at(output, x);

			lowpass1CCopy[j].compute_at(output, x);
			lowpass2CCopy[j].compute_at(output, x);
			lowpass1SCopy[j].compute_at(output, x);
			lowpass2SCopy[j].compute_at(output, x);
			lowpass1C[j].compute_at(output, x);
			lowpass2C[j].compute_at(output, x);
			lowpass1S[j].compute_at(output, x);
			lowpass2S[j].compute_at(output, x);

			phaseCCopy[j].compute_at(output, x);
			phaseSCopy[j].compute_at(output, x);
			phaseC[j].compute_at(output, x);
			phaseS[j].compute_at(output, x);
			phi[j].compute_at(output, x);

			r1Pyramid[j].compute_at(output, x);
			r2Pyramid[j].compute_at(output, x);

			lPyramidCopy[j].compute_at(output, x);
			lPyramid[j].compute_at(output, x);
			lPyramidUpX[j].compute_at(output, x);
		} else {
			outGPyramid[j].compute_root();
			outGPyramidUpX[j].compute_root();

			ampReg[j].compute_root().split(y, y, yi, 16);
			ampRegX[j].compute_at(ampReg[j], y);
			changeCReg[j].compute_root().split(y, y, yi, 16);
			changeCRegX[j].compute_at(changeCReg[j], y);
			changeSReg[j].compute_root().split(y, y, yi, 16);
			changeSRegX[j].compute_at(changeSReg[j], y);
			amp[j].compute_root();

			changeCTuple[j].compute_root();
			changeSTuple[j].compute_root();

			lowpass1CCopy[j].compute_root();
			lowpass2CCopy[j].compute_root();
			lowpass1SCopy[j].compute_root();
			lowpass2SCopy[j].compute_root();
			lowpass1C[j].compute_root();
			lowpass2C[j].compute_root();
			lowpass1S[j].compute_root();
			lowpass2S[j].compute_root();

			phaseCCopy[j].compute_root();
			phaseSCopy[j].compute_root();
			phaseC[j].compute_root();
			phaseS[j].compute_root();
			phi[j].compute_root();

			r1Pyramid[j].compute_root();
			r2Pyramid[j].compute_root();

			lPyramidCopy[j].compute_root();
			lPyramid[j].compute_root().split(y, y, yi, 8);
			lPyramidUpX[j].compute_at(lPyramid[j], y);
		}

		if (j>0) {
			gPyramid[j].compute_root().split(y, y, yi, 8);
			gPyramidDownX[j].compute_at(gPyramid[j], y);
		}

		if (j<=4) {
			outGPyramid[j].parallel(y, 4).vectorize(x, VECTOR_SIZE);
			outGPyramidUpX[j].parallel(y, 4).vectorize(x, VECTOR_SIZE);

			ampReg[j].parallel(y, 4).vectorize(x, VECTOR_SIZE);
			ampRegX[j].parallel(y, 4).vectorize(x, VECTOR_SIZE);
			changeCReg[j].parallel(y, 4).vectorize(x, VECTOR_SIZE);
			changeCRegX[j].parallel(y, 4).vectorize(x, VECTOR_SIZE);
			changeSReg[j].parallel(y, 4).vectorize(x, VECTOR_SIZE);
			changeSRegX[j].parallel(y, 4).vectorize(x, VECTOR_SIZE);
			amp[j].parallel(y, 4).vectorize(x, VECTOR_SIZE);

			changeCTuple[j].parallel(y, 4).vectorize(x, VECTOR_SIZE);
			changeSTuple[j].parallel(y, 4).vectorize(x, VECTOR_SIZE);

			lowpass1C[j].parallel(y, 4).vectorize(x, VECTOR_SIZE);
			lowpass2C[j].parallel(y, 4).vectorize(x, VECTOR_SIZE);
			lowpass1S[j].parallel(y, 4).vectorize(x, VECTOR_SIZE);
			lowpass2S[j].parallel(y, 4).vectorize(x, VECTOR_SIZE);

			phaseC[j].parallel(y, 4).vectorize(x, VECTOR_SIZE);
			phaseS[j].parallel(y, 4).vectorize(x, VECTOR_SIZE);
			phi[j].parallel(y, 4).vectorize(x, VECTOR_SIZE);

			r1Pyramid[j].parallel(y, 4).vectorize(x, VECTOR_SIZE);
			r2Pyramid[j].parallel(y, 4).vectorize(x, VECTOR_SIZE);

			lPyramid[j].parallel(y, 4).vectorize(x, VECTOR_SIZE);
			lPyramidUpX[j].parallel(y, 4).vectorize(x, VECTOR_SIZE);

			if (j>0) {
				gPyramid[j].parallel(y, 4).vectorize(x, VECTOR_SIZE);
				gPyramidDownX[j].parallel(y, 4).vectorize(x, VECTOR_SIZE);
			}
		}
	}

	// The final level
	gPyramid[pyramidLevels].compute_root().parallel(y, 4).vectorize(x, VECTOR_SIZE);
#endif
}

void RieszMagnifier::compileJIT(bool tile) {
	Target target = get_target_from_environment();
    schedule(tile);
	cerr << "Compiling JIT for target " << target.to_string() << " .. ";
	output.compile_jit();
	cerr << " done\n" << endl;
}

void RieszMagnifier::bindJIT(
        float a1,
        float a2,
        float b0,
        float b1,
        float b2,
        float alpha,
        int stabilize,
        vector<Image<float>> historyBuffer,
        vector<Image<float>> amplitudeBuffer
        )
{
	this->a1.set(a1);
	this->a2.set(a2);
	this->b0.set(b0);
	this->b1.set(b1);
	this->b2.set(b2);
	this->alpha.set(alpha);
	this->stabilize.set(stabilize);
	for (int j=0; j<pyramidLevels; j++) {
		this->historyBuffer  [j].set(historyBuffer[j]);
		this->amplitudeBuffer[j].set(amplitudeBuffer[j]);
    }
}

void RieszMagnifier::process(Buffer frame, Buffer out)
{
	pParam.set(frameCounter % CIRCBUFFER_SIZE);
	input.set(frame);
	output.realize(out);
	frameCounter++;
}

void RieszMagnifier::compute_ref_amplitude(Buffer frame, vector<Image<float>> amplitudeBuff)
{
    int width = frame.extent(0);
    int height= frame.extent(1);

    input.set(frame);

    cerr << "Computing reference amplitude ";
    for (int i=0; i<pyramidLevels; i++) {
        amp_orig[i].realize(amplitudeBuff[i]);
        cerr << ".";
    }
    cerr << " done" << endl;
}

void RieszMagnifier::computeBandSigmas() {
	for (int j=0; j<pyramidLevels; j++) {
		bandSigma[j] = 3;
	}
}

int RieszMagnifier::getPyramidLevels() {
    return pyramidLevels;
}
