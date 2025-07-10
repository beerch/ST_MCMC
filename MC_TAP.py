from functools import partial
import jax
import jax.numpy as jnp
from jaxopt import AndersonAcceleration
import optax

def _gamma(x, beta, r):
    """See Eq. (39)."""
    return jnp.sqrt(1 + beta**2 * jnp.sum(x**2, axis=-1, keepdims=True) / r**2)


def _phi(theta, beta, r):
    """See Eq. (38)."""
    return beta / (1 + _gamma(theta, beta, r)) * theta


def update_naive_mf(m0, _, x, J, beta, r):
    """See Eq. (47)."""
    theta = x + jnp.einsum("i j, j d -> i d", J, m0)
    m1 = _phi(theta, beta, r)
    return m1, m0


def _inv_phi(m, beta, r):
    """See Eq. (64)."""
    return 2 * r**2 / (beta * (r**2 - jnp.sum(m**2, axis=-1, keepdims=True))) * m


def _d2_m_d_alpha_2(m1, m0, x, J, beta, r):
    """See Eq. (58)."""
    g0 = _gamma(_inv_phi(m0, beta, r), beta, r)
    g1 = _gamma(_inv_phi(m1, beta, r), beta, r)
    v = -_inv_phi(m1, beta, r) + x + jnp.einsum("i j, j d -> i d", J, m0)

    return (
        (beta**2 * (1 + 3 * g1))
        / (r**4 * g1**3)
        * (
            jnp.einsum("i d, i d -> i", m1, v)[:, None] ** 2
            + jnp.einsum(
                "i j, i d -> i d",
                J**2,
                jnp.sum(m1**2, axis=-1, keepdims=True),
            )
            / (1 + g0)
            - jnp.einsum(
                "i j, i d, j d, i e, j e -> i",
                J**2,
                m1,
                m0,
                m1,
                m0,
            )[:, None]
            / (r**2 * g0)
        )
        * m1
        - (beta**2)
        / (r**2 * (g1**2 + g1))
        * (
            jnp.sum(v**2, axis=-1, keepdims=True)
            + jnp.einsum(
                "i j, j -> i",
                J**2,
                r**2 - jnp.sum(m0**2, axis=-1),
            )[:, None]
        )
        * m1
        - 2.0
        * beta**2
        / (r**2 * (g1**2 + g1))
        * (
            jnp.einsum("i d, i d, i f -> i f", v, m1, v)
            + jnp.einsum("i j, i d -> i d", J**2, m1 / (1 + g0))
            - jnp.einsum(
                "i j, i d, j d, j f -> i f",
                J**2,
                m1,
                m0,
                m0,
            )
            / (r**2 * g0)
        )
    )


def _f(m1, m0, x, J, beta, r):
    """See Eq. (61)."""
    g1 = _gamma(_inv_phi(m1, beta, r), beta, r)
    d2_m_d_alpha_2 = _d2_m_d_alpha_2(m1, m0, x, J, beta, r)

    ff = (
        (1 + g1)
        / (2 * beta)
        * (
            d2_m_d_alpha_2
            + (
                jnp.einsum("i d, i d -> i", m1, d2_m_d_alpha_2)[:, None]
                / ((r**2 * g1) / (1 + g1) - jnp.sum(m1**2, axis=-1, keepdims=True))
                * m1
            )
        )
    )
    return x + jnp.einsum("i j, j d -> i d", J, m0) + ff


def update_tap_mf(m0, _, x, J, beta, r):
    """See Eq. (65)."""

    def tap(m1, _m0, _x, _J, _beta, _r):
        return _phi(_f(m1, _m0, _x, _J, _beta, _r), _beta, _r)

    out = (
        AndersonAcceleration(
            fixed_point_fun=tap,
            tol=1e-3,
            maxiter=100,
        )
    ).run(_phi(x + J @ m0, beta, r), m0, x, J, beta, r)

    #jax.debug.print("{error}", error=out.state.error)

    return out.params, m0


def time_evolution(m0, steps, update_fun):
    final_carry, stacked_outputs = jax.lax.scan(update_fun, init=m0, xs=steps)
    return final_carry, stacked_outputs


def simulate(x, J, m0, steps, beta, r, update_fun=update_tap_mf):
    wrapped_time_evolution = partial(
        time_evolution,
        steps=steps,
        update_fun=partial(update_fun, x=x, J=J, beta=beta, r=r),
    )
    final_carry, stacked_outputs = jax.vmap(wrapped_time_evolution)(m0)
    return final_carry, stacked_outputs

def jaxtap(x, J, m0, steps, beta, r):
    _, stacked_outputs_tap = simulate(x, J, m0, steps, beta, r, update_fun=update_tap_mf)
    return stacked_outputs_tap.transpose((1, 0, 2, 3))

