import numpy as np

def expected_losses(posterior, loss):
    """
    posterior: (n_obs, n_states)
    loss: (n_states, n_actions)

    returns: (n_obs, n_actions)
    """
    posterior = np.asarray(posterior, dtype=float)
    loss = np.asarray(loss, dtype=float)

    return posterior @ loss


def bayes_policy(posterior, loss):
    """
    Returns the optimal action for each observation.
    """
    return np.argmin(expected_losses(posterior, loss), axis=1)


def covid_table(prevalence, sens, spec, drug_cost,
                qaly_young=60.0, qaly_old=10.0):

    # Posterior P(covid | negative)
    p_neg = ((1 - sens) * prevalence) / (
        (1 - sens) * prevalence + spec * (1 - prevalence)
    )

    # Posterior P(covid | positive)
    p_pos = (sens * prevalence) / (
        sens * prevalence + (1 - spec) * (1 - prevalence)
    )

    rows = []

    for test, p_covid in [(0, p_neg), (1, p_pos)]:
        for age, qaly in [(0, qaly_young), (1, qaly_old)]:

            cost_nothing = p_covid * qaly
            cost_drug = drug_cost

            action = 0 if cost_nothing <= cost_drug else 1

            rows.append(
                (
                    test,           # test result (0=neg,1=pos)
                    age,            # age group (0=young,1=old)
                    p_covid,
                    cost_nothing,
                    drug_cost,
                    action
                )
            )

    return rows