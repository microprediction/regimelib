Intensity basket
================

``rl.SwitchingIntensityBasket(models)``: several default intensities, each a ``SwitchingVasicek`` or
``SwitchingCoxIngersollRoss``, driven by one chain with independent diffusions.

The model
---------

.. math::

    d\lambda^k_t = a_k\,(b^k_{y_t} - \lambda^k_t)\,dt + \sigma^k_{y_t}\,dW^k_t, \qquad k = 1, \dots, K,

with one regime :math:`y_t` shared by all names. The common regime is the only source of dependence.

Reduction
---------

Given the regime path the names are independent, so the joint survival probability is the bond of the sum of the
intensities:

.. math::

    \mathbb{P}(\text{no default by } T \mid y_0 = i) = \mathbb{E}\big[e^{-\int_0^T\sum_k\lambda^k}\big]
      = e^{-\sum_kB_k(T)\,\lambda^k_0}\,a_i(T), \qquad g_i = \sum_k g^k_i .

The forcings add and the prefactors multiply. The product of the marginal survivals has the forcings solved
separately, and the difference between the two is the dependence the regime creates.

Closed form
-----------

For two Vasicek names with a common reversion speed the half-differences add,
:math:`\tilde g = \tilde g^1 + \tilde g^2`, and the Green–Kubo term of the joint survival contains the cross term
:math:`\varepsilon\int\tilde g^1\tilde g^2`, which the product of the marginals lacks. To first order

.. math::

    \log\frac{Q_{12}(T)}{Q_1(T)\,Q_2(T)} = \varepsilon\int_0^T\tilde g^1\tilde g^2 + O(\varepsilon^2),
    \qquad \int_0^T\tilde g^1\tilde g^2 = a^2\tilde b^1\tilde b^2I_2 - \tfrac12a\big(\tilde b^1\tilde s^2 + \tilde b^2\tilde s^1\big)I_3 + \tfrac14\tilde s^1\tilde s^2I_4 .

It is positive when the same regime raises both intensities and negative when it raises one and lowers the other.

Python
------

.. code-block:: python

    x0, a, sig, lam, T = 0.02, 0.5, 0.003, 5.0, 5.0
    chain = rl.RegimeChain.twoState(lam, lam)
    name = lambda levels: rl.SwitchingVasicek(chain, x0, a, levels, sig)
    E, B, I1, I2, I3, I4 = powers_of_B(a, T)

    for label, b1, b2 in (("same regime raises both", [0.06, 0.01], [0.06, 0.01]),
                          ("regime raises one, lowers the other", [0.06, 0.01], [0.01, 0.06])):
        closed = (1 / lam) * a * a * half(b1)[1] * half(b2)[1] * I2          # sigma does not switch here
        n1, n2 = name(b1), name(b2)
        joint = library_bond(rl.SwitchingIntensityBasket([n1, n2]), T)
        library = np.log(joint / (library_bond(n1, T) * library_bond(n2, T)))
        print(f"{label}:")
        print(f"  closed form, first order   {closed:+.6f}")
        print(f"  library, numerical         {library:+.6f}")
        print(f"  default correlation        {rl.SwitchingIntensityBasket([n1, n2]).defaultCorrelation(T):+.4f}")


.. code-block:: text

    same regime raises both:
      closed form, first order   +0.000290
      library, numerical         +0.000280
      default correlation        +0.0017
    regime raises one, lowers the other:
      closed form, first order   -0.000290
      library, numerical         -0.000279
      default correlation        -0.0018

``rl.FirstToDefaultSwap`` on the basket is a single-name credit default swap on the summed intensity.
