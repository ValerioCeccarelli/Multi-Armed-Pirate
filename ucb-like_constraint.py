import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

# the agent does not need the values of the available bids or the valuation
# the first bid is 0
class UCBLikeAgent():
    def __init__(self, K, B, T, range=1):
        self.K = K
        self.T = T
        self.range = range
        self.a_t = None # it's an index, not the actual bid
        self.avg_f = np.zeros(K)
        self.avg_c = np.zeros(K)
        self.N_pulls = np.zeros(K)
        self.budget = B
        self.rho = B/T
        self.t = 0
        self.budget_over_time = np.zeros(T)
    
    def pull_arm(self):
        """Pulls an arm based on UCB-like strategy with constraints; if budget is exhausted, returns -1."""
        if self.budget < 1:
            self.a_t = -1
        elif self.t < self.K:
            self.a_t = self.t 
        else:
            f_ucbs = self.avg_f + self.range*np.sqrt(2*np.log(self.t+1)/self.N_pulls)
            c_lcbs = self.avg_c - self.range*np.sqrt(2*np.log(self.t+1)/self.N_pulls)
            gamma_t = self.compute_opt(f_ucbs, c_lcbs)
            self.a_t = np.random.choice(self.K, p=gamma_t)

        return self.a_t

    def compute_opt(self, f_ucbs, c_lcbs):
        if np.sum(c_lcbs <= np.zeros(len(c_lcbs))):
            gamma = np.zeros(len(f_ucbs))
            gamma[np.argmax(f_ucbs)] = 1
            return gamma
        c = -f_ucbs
        A_ub = [c_lcbs]
        b_ub = [self.rho]
        A_eq = [np.ones(self.K)]
        b_eq = [1]
        res = optimize.linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=(0,1))
        gamma = res.x
        return gamma
    
    def update(self, f_t, c_t):
        if int(c_t) not in [0, 1]:
            raise ValueError(f"c_t invalid, Received: {c_t}")
        if self.a_t != -1:  # if an arm was pulled
            self.N_pulls[self.a_t] += 1
            self.avg_f[self.a_t] += (f_t - self.avg_f[self.a_t])/self.N_pulls[self.a_t]
            self.avg_c[self.a_t] += (c_t - self.avg_c[self.a_t])/self.N_pulls[self.a_t]
            self.budget -= c_t
            self.budget_over_time[self.t] = self.budget
        self.t += 1

def make_gaussian_sampler(mean=0.0, std_dev=1.0):
    def func(_N):
        return np.random.normal(loc=mean, scale=std_dev, size=_N)
    return func

class PricingEnvironment:
    def __init__(self, T: int, sampler):
        self.T = T
        self.values = sampler(T)
        
    def round(self, p_t, t) -> bool:
        """Returns True if the buyer accepts the price at time t."""
        v_t = self.values[t]
        return v_t >= p_t
    
buyer_value = 12
buyer_indecision = 3
buyer_sampler = make_gaussian_sampler(mean=buyer_value, std_dev=buyer_indecision)

# T = 1000 # Questo ci piace
# K = 100
# B = 900

T = 1000 # Questo ci piace
K = 100
B = 100

prices = np.linspace(0, 20, K)

def compute_optimal_price(prices, env, B):
    rewards = []
    for p in prices:
        reward = sum([p for t in range(T) if env.round(p, t)][:B])
        rewards.append(reward)

    best_arm = np.argmax(rewards)
    print(f"Best price: {prices[best_arm]} (arm {best_arm}), Total reward: {rewards[best_arm]}")

    best_price_index = best_arm
    best_price = prices[best_price_index]

    return best_price, best_price_index

regret_per_trial = []


n_trials = 10
for seed in range(n_trials):
    np.random.seed(seed)
    agent_regrets = np.array([])
    budget_finito_my = float("inf")
    budget_finito_opt = float("inf")

    agent = UCBLikeAgent(K=K, 
                     B=B,
                     T=T,
                     range=1) # TODO: vedi meglio questo range

    env = PricingEnvironment(T=T, sampler=buyer_sampler)
    best_price, best_price_index = compute_optimal_price(prices, env, B)

    best_remaining_budget = B

    for u in range(T):

        my_p_index = agent.pull_arm()
        if my_p_index == -1:
            my_r_t = 0
            budget_finito_my = min(budget_finito_my, u)
        else:
            my_p = prices[my_p_index]

            my_vinto = env.round(my_p, u)
            my_r_t = my_p if my_vinto else 0

            my_c_t = 1 if my_vinto else 0
        
        agent.update(my_r_t, my_c_t)

        if best_remaining_budget > 0:
            best_vinto = env.round(best_price, u)
            best_r_t = best_price if best_vinto else 0
            best_remaining_budget -= 1 if best_vinto else 0
        else:
            budget_finito_opt = min(budget_finito_opt, u)
            best_r_t = 0
        
        regret = best_r_t - my_r_t
        agent_regrets = np.append(agent_regrets, regret)

    print(f"Budget finito at u={budget_finito_my}, optimal at u={budget_finito_opt}")
    cumulative_regret = np.cumsum(agent_regrets)
    regret_per_trial.append(cumulative_regret)

regret_per_trial = np.array(regret_per_trial)

average_regret = regret_per_trial.mean(axis=0)
regret_sd = regret_per_trial.std(axis=0)

print(f"Budget finito at u={budget_finito_my}")

# plt.plot(np.arange(T), average_regret, label='Average Regret')
# plt.title('cumulative regret of UCB1')
# plt.fill_between(np.arange(T),
#                 average_regret-regret_sd/np.sqrt(n_trials),
#                 average_regret+regret_sd/np.sqrt(n_trials),
#                 alpha=0.3,
#                 label='Uncertainty')
# #plt.plot((0,T-1), (average_regret[0], average_regret[-1]), 'ro', linestyle="--")
# plt.xlabel('$t$')
# plt.legend()
# plt.show()


# # Plotting the budget over time
# plt.figure()
# plt.plot(np.arange(T), agent.budget_over_time, label='Budget over time')
# plt.axhline(0, color='red', linestyle='--', label='Budget exhausted')
# plt.xlabel('Time step')
# plt.ylabel('Remaining Budget')
# plt.title('Budget over time')
# plt.legend()
# plt.show()

fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

# Subplot 1: Cumulative Regret
axs[0].plot(np.arange(T), average_regret, label='Average Regret')
axs[0].fill_between(np.arange(T),
                    average_regret - regret_sd / np.sqrt(n_trials),
                    average_regret + regret_sd / np.sqrt(n_trials),
                    alpha=0.3,
                    label='Uncertainty')
axs[0].set_title('Cumulative Regret of UCB1')
axs[0].set_ylabel('Cumulative Regret')
axs[0].legend()
axs[0].grid(True)

# Subplot 2: Budget over Time
axs[1].plot(np.arange(T), agent.budget_over_time, label='Budget over time')
axs[1].axhline(0, color='red', linestyle='--', label='Budget exhausted')
axs[1].set_title('Budget Over Time')
axs[1].set_xlabel('Time Step')
axs[1].set_ylabel('Remaining Budget')
axs[1].legend()
axs[1].grid(True)

plt.tight_layout()
plt.show()



# plt.figure()
# plt.barh(prices, agent.N_pulls)
# plt.axhline(best_price, color='red', label='Best price')
# plt.ylabel('prices')
# plt.xlabel('numer of pulls')
# plt.legend()
# plt.title('Number of pulls for each action')
# plt.show()