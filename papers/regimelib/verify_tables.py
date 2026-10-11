"""Generates tables.tex for the regimelib paper from actual runs: frozen-limit agreement with QuantLib, the
expansion's error by order and holding time, engine timings, and parameter greeks against finite differences."""
import math, time, warnings
import numpy as np
import QuantLib as ql
import regimelib as rl
warnings.simplefilter("ignore")
REF = ql.Date(1, 1, 2020); ql.Settings.instance().evaluationDate = REF; dc = ql.Actual365Fixed(); cal = ql.NullCalendar()
rts = lambda r: ql.YieldTermStructureHandle(ql.FlatForward(REF, r, dc))
out = []

# ---------------------------------------------------------------- frozen limit
rows = []
chain = rl.RegimeChain.twoState(3.0, 5.0)
def rel(a, b): return abs(a / b - 1)
def add(name, ours, ql_, engine):
    rows.append((name, engine, ours, ql_, rel(ours, ql_)))
# Vasicek bond
vas = ql.Vasicek(0.03, 0.5, 0.04, 0.012); m = rl.SwitchingVasicek(chain, 0.03, 0.5, 0.04, 0.012)
b = rl.ZeroCouponBond(5.0); b.setPricingEngine(rl.FastSwitchingEngine(m, order=2)); add("Vasicek bond, $T=5$", b.NPV(), vas.discountBond(0.0, 5.0, 0.03), "Vasicek::discountBond")
# HW bond option
ts = rts(0.03); hw = ql.HullWhite(ts, 0.5, 0.012); mh = rl.SwitchingHullWhite(chain, ts, 0.5, 0.012)
o = rl.ZeroCouponBondOption("call", 0.9, 2.0, 5.0); o.setPricingEngine(rl.FastSwitchingEngine(mh, order=2)); add("Hull--White bond call", o.NPV(), hw.discountBondOption(ql.Option.Call, 0.9, 2.0, 5.0), "HullWhite::discountBondOption")
# G2 bond option
g2 = ql.G2(ts, 0.5, 0.012, 0.08, 0.009, -0.6); mg = rl.SwitchingG2(chain, ts, 0.5, 0.012, 0.08, 0.009, -0.6)
o = rl.ZeroCouponBondOption("call", 0.9, 2.0, 5.0); o.setPricingEngine(rl.NumericalSwitchingEngine(mg)); add("G2++ bond call", o.NPV(), g2.discountBondOption(ql.Option.Call, 0.9, 2.0, 5.0), "G2::discountBondOption")
# swaption HW Jamshidian
index = ql.IborIndex("idx", ql.Period(1, ql.Years), 0, ql.USDCurrency(), cal, ql.Unadjusted, False, dc, ts)
start = REF + ql.Period(2, ql.Years); end = start + ql.Period(5, ql.Years)
sched = ql.Schedule(start, end, ql.Period(1, ql.Years), cal, ql.Unadjusted, ql.Unadjusted, ql.DateGeneration.Forward, False)
swap = ql.VanillaSwap(ql.VanillaSwap.Payer, 1.0, sched, 0.03, dc, sched, index, 0.0, dc)
sw = ql.Swaption(swap, ql.EuropeanExercise(start)); sw.setPricingEngine(ql.JamshidianSwaptionEngine(hw, ts))
fixed = [dc.yearFraction(REF, d) for d in list(sched)[1:]]; T0 = dc.yearFraction(REF, start)
s = rl.Swaption("payer", T0, fixed, 0.03); s.setPricingEngine(rl.FastSwitchingEngine(mh, order=2)); add("Hull--White payer swaption $2\\times5$", s.NPV(), sw.NPV(), "JamshidianSwaptionEngine")
# Bermudan HW
be = ql.Swaption(swap, ql.BermudanExercise(list(sched)[:-1])); be.setPricingEngine(ql.FdHullWhiteSwaptionEngine(hw, 400, 400))
sb = rl.Swaption("payer", T0, fixed, 0.03, exerciseTimes=[dc.yearFraction(REF, d) for d in list(sched)[:-1]]); sb.setPricingEngine(rl.SwitchingFDEngine(mh, n=1201, steps=600)); add("Hull--White Bermudan swaption", sb.NPV(), be.NPV(), "FdHullWhiteSwaptionEngine")
# cap
sched2 = ql.Schedule(REF + ql.Period(1, ql.Years), REF + ql.Period(5, ql.Years), ql.Period(1, ql.Years), cal, ql.Unadjusted, ql.Unadjusted, ql.DateGeneration.Forward, False)
cap = ql.Cap(ql.IborLeg([1.0], sched2, index, dc), [0.03]); cap.setPricingEngine(ql.AnalyticCapFloorEngine(hw, ts))
c = rl.CapFloor("cap", [dc.yearFraction(REF, d) for d in list(sched2)], 0.03); c.setPricingEngine(rl.FastSwitchingEngine(mh, order=2)); add("Hull--White cap, 4 caplets", c.NPV(), cap.NPV(), "AnalyticCapFloorEngine")
# BS vanilla, Heston, Bates
S0, r, q, K = 100.0, 0.03, 0.01, 105.0
bsp = ql.BlackScholesMertonProcess(ql.QuoteHandle(ql.SimpleQuote(S0)), rts(q), rts(r), ql.BlackVolTermStructureHandle(ql.BlackConstantVol(REF, cal, 0.25, dc)))
ex1 = ql.EuropeanExercise(REF + ql.Period(365, ql.Days)); pay = ql.PlainVanillaPayoff(ql.Option.Put, K)
v = ql.VanillaOption(pay, ex1); v.setPricingEngine(ql.AnalyticEuropeanEngine(bsp))
mb = rl.SwitchingBlackScholesProcess(chain, S0, r, q, 0.25); ob = rl.VanillaOption(pay, ex1); ob.setPricingEngine(rl.FastSwitchingEngine(mb, order=2)); add("Black--Scholes put", ob.NPV(), v.NPV(), "AnalyticEuropeanEngine")
hp = ql.HestonProcess(rts(r), rts(q), ql.QuoteHandle(ql.SimpleQuote(S0)), 0.04, 1.5, 0.05, 0.4, -0.5); v.setPricingEngine(ql.AnalyticHestonEngine(ql.HestonModel(hp)))
mhes = rl.SwitchingHestonModel(chain, S0, r, q, 0.04, 1.5, 0.05, 0.4, -0.5); ob.setPricingEngine(rl.NumericalSwitchingEngine(mhes)); add("Heston put", ob.NPV(), v.NPV(), "AnalyticHestonEngine")
# American, barrier, Asian
am = ql.VanillaOption(pay, ql.AmericanExercise(REF, REF + ql.Period(365, ql.Days))); am.setPricingEngine(ql.FdBlackScholesVanillaEngine(bsp, 2000, 2000))
oa = rl.VanillaOption(pay, ql.AmericanExercise(REF, REF + ql.Period(365, ql.Days))); oa.setPricingEngine(rl.SwitchingFDEngine(mb, n=1601, steps=800)); add("American put", oa.NPV(), am.NPV(), "FdBlackScholesVanillaEngine")
bo = ql.BarrierOption(ql.Barrier.DownOut, 85.0, 0.0, ql.PlainVanillaPayoff(ql.Option.Call, 100.0), ex1); bo.setPricingEngine(ql.AnalyticBarrierEngine(bsp))
ob2 = rl.BarrierOption(ql.Barrier.DownOut, 85.0, 0.0, ql.PlainVanillaPayoff(ql.Option.Call, 100.0), ex1); ob2.setPricingEngine(rl.SwitchingFDEngine(mb, n=2001, steps=800)); add("Down-and-out call", ob2.NPV(), bo.NPV(), "AnalyticBarrierEngine")
asq = ql.ContinuousAveragingAsianOption(ql.Average.Geometric, ql.PlainVanillaPayoff(ql.Option.Call, 100.0), ex1); asq.setPricingEngine(ql.AnalyticContinuousGeometricAveragePriceAsianEngine(bsp))
oas = rl.ContinuousGeometricAsianOption(ql.PlainVanillaPayoff(ql.Option.Call, 100.0), ex1); oas.setPricingEngine(rl.NumericalSwitchingEngine(mb)); add("Geometric Asian call", oas.NPV(), asq.NPV(), "AnalyticContinuousGeometric...")
# hybrid
hwm = ql.HullWhite(ts, 0.4, 0.015); vh = ql.VanillaOption(pay, ql.EuropeanExercise(REF + ql.Period(730, ql.Days))); vh.setPricingEngine(ql.AnalyticBSMHullWhiteEngine(0.4, bsp, hwm))
mhy = rl.SwitchingEquityRates(rl.SwitchingBlackScholesProcess(chain, S0, r, q, 0.25), rl.SwitchingHullWhite(chain, ts, 0.4, 0.015), rho=0.4)
oh = rl.VanillaOption(pay, ql.EuropeanExercise(REF + ql.Period(730, ql.Days))); oh.setPricingEngine(rl.NumericalSwitchingEngine(mhy)); add("Black--Scholes--Hull--White put, $\\rho=0.4$", oh.NPV(), vh.NPV(), "AnalyticBSMHullWhiteEngine")
# CDS
mi = rl.SwitchingCoxIngersollRoss(chain, 0.02, 0.03, 0.5, 0.08); eng = rl.NumericalSwitchingEngine(mi)
fine = [i / 48 for i in range(1, 289)]; probs = [1.0]
for t in fine:
    bb = rl.ZeroCouponBond(t); bb.setPricingEngine(eng); probs.append(bb.NPV())
curve = ql.DefaultProbabilityTermStructureHandle(ql.SurvivalProbabilityCurve([REF] + [REF + ql.Period(int(round(t * 365)), ql.Days) for t in fine], probs, dc, cal))
sched3 = ql.Schedule(REF, REF + ql.Period(5, ql.Years), ql.Period(6, ql.Months), cal, ql.Unadjusted, ql.Unadjusted, ql.DateGeneration.Forward, False)
qc = ql.CreditDefaultSwap(ql.Protection.Buyer, 1.0, 0.02, sched3, ql.Unadjusted, dc, True, True); qc.setPricingEngine(ql.MidPointCdsEngine(curve, 0.4, rts(0.03)))
cds = rl.CreditDefaultSwap("buyer", 0.02, [0.5 * i for i in range(1, 11)], 0.4, discount=0.03); cds.setPricingEngine(eng); cds.NPV(); add("CDS fair spread, CIR intensity", cds.fairSpread(), qc.fairSpread(), "MidPointCdsEngine")
import io
f = io.StringIO()
if True:
    f.write("\\begin{tabular}{llrrr}\\toprule\nInstrument & QuantLib engine & regimelib & QuantLib & rel.\\ diff.\\\\\\midrule\n")
    for name, engine, a, b_, e in rows:
        f.write(f"{name} & \\texttt{{{engine}}} & {a:.8g} & {b_:.8g} & {e:.1e}\\\\\n")
    f.write("\\bottomrule\\end{tabular}\n")
    # ---------------------------------------------------------------- expansion by order and holding time
    f.write("\n%% orders\n\\begin{tabular}{lrrrrrr}\\toprule\n$\\varepsilon$ & order 0 & 1 & 2 & 3 & 4 & 6\\\\\\midrule\n")
    for lam in (20.0, 10.0, 5.0, 3.0):
        ch = rl.RegimeChain.twoState(lam, lam); eps = ch.meanHoldingTime()
        mv = rl.SwitchingVasicek(ch, 0.03, 0.5, [0.06, 0.02], [0.015, 0.008])
        swv = rl.Swaption("payer", 2.0, [3.0, 4.0, 5.0, 6.0, 7.0], 0.035, notional=100.0)
        swv.setPricingEngine(rl.NumericalSwitchingEngine(mv)); ref = swv.NPV(); errs = []
        for order in (0, 1, 2, 3, 4, 6):
            swv.setPricingEngine(rl.FastSwitchingEngine(mv, order=order))
            try:
                errs.append(f"{abs(swv.NPV() / ref - 1):.1e}")
            except ArithmeticError:                       # the Gil-Pelaez rule refuses a diverged expansion
                errs.append("diverged")
        f.write(f"{eps:.3f} & " + " & ".join(errs) + "\\\\\n")
    f.write("\\bottomrule\\end{tabular}\n")
    # ---------------------------------------------------------------- timings
    f.write("\n%% timings\n\\begin{tabular}{lrrr}\\toprule\nInstrument (switching on) & expansion, order 4 & numerical & grid / Monte Carlo\\\\\\midrule\n")
    ch = rl.RegimeChain.twoState(12.0, 8.0)
    def tm(inst, eng):
        inst.setPricingEngine(eng); t = time.time(); inst.NPV(); return time.time() - t
    mv = rl.SwitchingVasicek(ch, 0.03, 0.5, [0.06, 0.02], [0.015, 0.008]); bond = rl.ZeroCouponBond(5.0)
    f.write(f"Vasicek bond & {tm(bond, rl.FastSwitchingEngine(mv, order=4))*1e3:.1f} ms & {tm(bond, rl.NumericalSwitchingEngine(mv))*1e3:.1f} ms & {tm(bond, rl.MonteCarloSwitchingEngine(mv, paths=100000))*1e3:.0f} ms (MC, $10^5$ paths)\\\\\n")
    swv = rl.Swaption("payer", 2.0, [3.0, 4.0, 5.0, 6.0, 7.0], 0.035, notional=100.0)
    f.write(f"Vasicek swaption & {tm(swv, rl.FastSwitchingEngine(mv, order=4))*1e3:.0f} ms & {tm(swv, rl.NumericalSwitchingEngine(mv))*1e3:.0f} ms & {tm(swv, rl.SwitchingFDEngine(mv, n=1201, steps=400))*1e3:.0f} ms (grid)\\\\\n")
    mb2 = rl.SwitchingBlackScholesProcess(ch, 100.0, 0.03, 0.0, [0.35, 0.15]); ov = rl.VanillaOption(("call", 100.0), maturity=1.0)
    oam = rl.VanillaOption(("put", 100.0), exercise="american", maturity=1.0)
    f.write(f"Black--Scholes call / American put & {tm(ov, rl.FastSwitchingEngine(mb2, order=4))*1e3:.0f} ms & {tm(ov, rl.NumericalSwitchingEngine(mb2))*1e3:.0f} ms & {tm(oam, rl.SwitchingFDEngine(mb2, n=1601, steps=600))*1e3:.0f} ms (grid, American)\\\\\n")
    mh2 = rl.SwitchingHestonModel(ch, 100.0, 0.03, 0.0, 0.04, 1.5, [0.08, 0.03], 0.4, -0.5)
    f.write(f"Heston call & {tm(ov, rl.FastSwitchingEngine(mh2, order=4))*1e3:.0f} ms & {tm(ov, rl.NumericalSwitchingEngine(mh2))*1e3:.0f} ms & --\\\\\n")
    f.write("\\bottomrule\\end{tabular}\n")
    # ---------------------------------------------------------------- parameter greeks
    from regimelib.symbolic import VasicekBondFirstOrder
    f.write("\n%% greeks\n\\begin{tabular}{lrrr}\\toprule\nParameter & formula & finite difference & rel.\\ diff.\\\\\\midrule\n")
    Q = [[-5.0, 3.0, 2.0], [4.0, -9.0, 5.0], [1.0, 6.0, -7.0]]; kappa_, thetas, sigmas, r0_, T_ = 0.5, [0.08, 0.05, 0.01], [0.015, 0.01, 0.006], 0.03, 4.0
    fo = VasicekBondFirstOrder(); params = dict(r0=r0_, kappa=kappa_, T=T_, thetas=thetas, sigmas=sigmas)
    def price(Qm, th, sg):
        mm = rl.SwitchingVasicek(rl.RegimeChain(Qm), r0_, kappa_, th, sg); bb = rl.ZeroCouponBond(T_); bb.setPricingEngine(rl.FastSwitchingEngine(mm, order=1, regime=1)); return bb.NPV()
    h = 1e-5
    for wrt, label in ((("theta", 0), "$\\partial P/\\partial\\theta_1$"), (("sigma", 1), "$\\partial P/\\partial\\sigma_2$"), (("q", 0, 1), "$\\partial P/\\partial Q_{12}$"), (("q", 2, 1), "$\\partial P/\\partial Q_{32}$")):
        g = fo.parameterGreek(rl.RegimeChain(Q), wrt, regime=1, **params)
        def bumped(eps):
            Qm = [row[:] for row in Q]; th = list(thetas); sg = list(sigmas)
            if wrt[0] == "q": Qm[wrt[1]][wrt[2]] += eps; Qm[wrt[1]][wrt[1]] -= eps
            elif wrt[0] == "theta": th[wrt[1]] += eps
            else: sg[wrt[1]] += eps
            return price(Qm, th, sg)
        fd = (bumped(h) - bumped(-h)) / (2 * h)
        f.write(f"{label} & {g:.8g} & {fd:.8g} & {abs(g/fd-1):.1e}\\\\\n")
    f.write("\\bottomrule\\end{tabular}\n")
text = f.getvalue()
parts = text.split("\n%% ")
names = ["frozen", "orders", "timings", "greeks"]
for name, part in zip(names, parts):
    body = part.split("\n", 1)[1] if name != "frozen" else part
    open(f"tables_{name}.tex", "w").write(body)
print(text)
