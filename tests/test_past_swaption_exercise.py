"""Issue #341: refuse seasoned exercise schedules, without removing today's right."""
import pytest
import QuantLib as ql
import regimelib as rl


@pytest.mark.parametrize('kind', ['payer', 'receiver'])
@pytest.mark.parametrize('times', [[-1., .5], [-2., -1., .5], [-2., -1.], [-1e-14, .5]])
def test_negative_numeric_exercises_are_refused(kind, times):
    with pytest.raises(ValueError, match='nonnegative.*seasoned'):
        rl.Swaption(kind, times[0], [.5, 1.5], .03, exerciseTimes=times)


def test_past_date_schedule_is_refused():
    settings = ql.Settings.instance()
    previous = settings.evaluationDate
    try:
        settings.evaluationDate = ql.Date(31, 10, 2026)
        with pytest.raises(ValueError, match='nonnegative.*seasoned'):
            rl.Swaption('payer', ql.BermudanExercise([ql.Date(31, 10, 2025), ql.Date(30, 4, 2027)]),
                        [ql.Date(30, 4, 2027), ql.Date(30, 4, 2028)], .03, notional=100_000_000.)
    finally:
        settings.evaluationDate = previous


def test_negative_european_maturity_is_refused():
    with pytest.raises(ValueError, match='nonnegative.*seasoned'):
        rl.Swaption('receiver', -1., [.5, 1.5], .03)


@pytest.mark.parametrize('information', ['observed', 'inferred'])
@pytest.mark.parametrize('sigma', [0., .01])
@pytest.mark.parametrize('field', ['maturity', 'exerciseTimes'])
def test_fd_refuses_mutated_past_times_before_shortcuts(information, sigma, field):
    option = rl.Swaption('payer', 0., [.5, 1.5], .03, exerciseTimes=[0., .5])
    setattr(option, field, -1. if field == 'maturity' else [-1., .5])
    model = rl.SwitchingVasicek(rl.RegimeChain.twoState(3., 5.), .03, .5, [.03, .04], sigma)
    engine = rl.SwitchingFDEngine(model, information=information, n=101, steps=20)
    with pytest.raises(ValueError, match='nonnegative.*seasoned'):
        engine.calculate(option)


@pytest.mark.parametrize('kind', ['payer', 'receiver'])
def test_future_only_frozen_control_matches_quantlib(kind):
    T, S, N, K = 181/365, 547/365, 100_000_000., .03
    C = N*(1 + K*(S-T))
    expected = C*ql.Vasicek(.03, .5, .03, .01).discountBondOption(
        ql.Option.Put if kind == 'payer' else ql.Option.Call, N/C, T, S)
    model = rl.SwitchingVasicek(rl.RegimeChain([[0.]]), .03, .5, .03, .01)
    option = rl.Swaption(kind, T, [S], K, notional=N)
    option.setPricingEngine(rl.SwitchingFDEngine(model, n=401, steps=400, width=(-.15, .20)))
    assert option.NPV() == pytest.approx(expected, rel=5e-4)
