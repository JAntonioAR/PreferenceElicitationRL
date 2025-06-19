import pandas as pd
import numpy as np
from pymoo.core.problem import ElementwiseProblem
from pymoo.core.repair import Repair
from pymoo.algorithms.moo.sms import SMSEMOA
from pymoo.optimize import minimize
from plotly.subplots import make_subplots
import plotly.graph_objs as go
import plotly.figure_factory as ff
import plotly.express as px
from IPython.display import clear_output
from sklearn.preprocessing import MinMaxScaler
import joblib
import scipy.optimize
import pymc3 as pm
import gym
from gym import spaces
from stable_baselines3.common.vec_env import DummyVecEnv


class PortfolioProblem(ElementwiseProblem):

    def __init__(self, returns_df, **kwargs):
        super().__init__(n_var=len(returns_df.columns), n_obj=2, xl=0.0, xu=1.0, **kwargs)
        self.returns_df = returns_df

    def _evaluate(self, x, out, *args, **kwargs):
        port_returns = self.returns_df @ x
        exp_return = port_returns.mean()
        exp_risk = port_returns.var()

        out["F"] = [exp_risk, -exp_return]

class PortfolioRepair(Repair):

    def _do(self, problem, X, **kwargs):
        X[X < 1e-3] = 0
        return X / X.sum(axis=1, keepdims=True)

def sample_time_interval(df, n):
    """Sample an interval of n timesteps of data
    
    Attributes
    ----------
        df: DataFrame
            input data
        n: int
            number of timesteps

    """
    end_idx = np.random.randint(n, len(df)-1)
    start_idx = end_idx - n
    return df.iloc[start_idx:end_idx]

def process_comparison_data(port_returns1, port_returns2, pareto_fr, pareto_fr_w):
    comp_data = {}

    comp_data["mean1"] = port_returns1.mean()
    comp_data["var1"] = port_returns1.var()

    comp_data["mean2"] = port_returns2.mean()
    comp_data["var2"] = port_returns2.var()

    comp_data["port_returns1"] = port_returns1.values
    comp_data["port_returns2"] = port_returns2.values
    comp_data["pareto_fr"] = pareto_fr
    comp_data["pareto_fr_w"] = pareto_fr_w

    return comp_data

def compute_comparison_data(train_df, method="MV"):
    """Display the empirical distribution of returns of 2 different portfolios
    
    
    """
    m = train_df.shape[1]

    #stock_returns = sample_time_interval(train_df, n)
    stock_returns = train_df.sample(1)["return_list"].values[0]

    if method == None:
        w1 = np.random.dirichlet((1.0,)*m)
        w2 = np.random.dirichlet((1.0,)*m)
    else:
        problem = PortfolioProblem(stock_returns)
        algorithm = SMSEMOA(repair=PortfolioRepair())
        res = minimize(problem,
                       algorithm,
                       verbose=False)
        
        X, F = res.opt.get("X", "F")
        F = F * [1, -1]

        np.random.shuffle(X)
        w1, w2 = X[:2]
        

    port_returns1 = stock_returns @ w1
    port_returns2 = stock_returns @ w2

    comparison_data = process_comparison_data(port_returns1, port_returns2, F, X)

    return comparison_data

def create_comparison_dataset(train_df, num_comparisons, comparison_dataset=None, method="MV"):
    if comparison_dataset == None:
        comparison_dataset = []
    
    for i in range(num_comparisons):
        comparison_dataset.append(compute_comparison_data(train_df, method))
    
    return pd.DataFrame(comparison_dataset)

def get_human_feedback(port_returns1, port_returns2):

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.7, 0.3], vertical_spacing=0.05)

    fig2 = ff.create_distplot([port_returns1, port_returns2],
                              ["Portfolio1", "Portfolio2"],
                              bin_size=0.005, show_rug=False)

    fig.add_trace(go.Histogram(fig2["data"][0]), row=1, col=1)
    fig.add_trace(go.Histogram(fig2["data"][1]), row=1, col=1)
    fig.add_trace(go.Scatter(fig2["data"][2]), row=1, col=1)
    fig.add_trace(go.Scatter(fig2["data"][3]), row=1, col=1)

    fig.add_trace(go.Box(x=port_returns1,
                         notched=True,
                         showlegend=False,
                         marker_color=fig2["data"][0].marker.color,
                         name=fig2["data"][0].name), row=2, col=1)
    fig.add_trace(go.Box(x=port_returns2,
                         notched=True,
                         showlegend=False,
                         marker_color=fig2["data"][1].marker.color,
                         name=fig2["data"][1].name), row=2, col=1)

    fig.update_layout(barmode='overlay')
    fig.update_yaxes(showticklabels=False, row=2, col=1)
    fig.update_xaxes(showgrid=True)

    fig.add_vline(x=port_returns1.mean(), line_width=1, line_color=fig2["data"][0].marker.color,
                  row=2, col=1)
    fig.add_vline(x=port_returns2.mean(), line_width=1, line_color=fig2["data"][1].marker.color,
                  row=2, col=1)

    fig.update_layout(margin=dict(l=20, r=20, t=20, b=20), template="simple_white", font=dict(family="Times New Roman", size=22.5),
                      legend=dict(font=dict(family="Times New Roman", size=17.5)))
    fig.show()

    while True:
        try:
            pref_portfolio = int(input('Preferred Portfolio:'))
        except ValueError:
            print('Please enter a valid integer (1 or 2)')
            continue
        if pref_portfolio not in [1,2]:
            print('Please enter a valid integer (1 or 2)')
            continue
        clear_output(wait=True)
        break

    return {"pref_portfolio":pref_portfolio, "comparison_fig":fig}

def get_preferred_portfolio(comparison_df, pref_model=None):
    preferred_portfolios = []

    for i in range(len(comparison_df)):
        comparison_data = comparison_df.iloc[i]

        if pref_model == None:
            port_returns1 = comparison_data.port_returns1
            port_returns2 = comparison_data.port_returns2
            human_feedback = get_human_feedback(port_returns1, port_returns2)
            pref_portfolio = human_feedback["pref_portfolio"] - 1
            fig = human_feedback["comparison_fig"]
        else:
            criteria = comparison_data.values[:-4].reshape((2,-1))
            pref_portfolio = int(pref_model(criteria[0]) <= pref_model(criteria[1]))

        preferred_portfolios.append(pref_portfolio)
    
    comparison_df["pref_portfolio"] =preferred_portfolios
    return preferred_portfolios if pref_model != None else {"pref_portfolios":preferred_portfolios, "last_comparison_fig":fig}

def data_split(df, start, end, target_date_col="date"):
    data = df[(df[target_date_col] >= start) & (df[target_date_col] < end)]
    data = data.sort_values([target_date_col, "tic"], ignore_index=True)
    data.index = data[target_date_col].factorize()[0]
    return data

def fit_pref_model(criteria, beta_samples):
    return beta_samples[:, 0]*criteria[0] - beta_samples[:, 1]*criteria[1]

def rebalance_classes(df, num_criteria):
    n = len(df)//2
    n0 = len(df[df.pref_portfolio == 0])
    n1 = len(df[df.pref_portfolio == 1])

    m, dominating_class, dominated_class = [n1 - n, 1, 0] if n1 >= n0 else [n0 - n, 0, 1]

    idx = df[df.pref_portfolio == dominating_class].sample(m).index
    df.iloc[idx, 0:num_criteria], df.iloc[idx, num_criteria:2*num_criteria] = df.iloc[idx, num_criteria:2*num_criteria], df.iloc[idx, 0:num_criteria]
    df.iloc[idx, -1] = dominated_class

def scale_feedback_df(feedback_df, name):
    scaler = MinMaxScaler()
    feedback_sc_df = pd.DataFrame()
    feedback_sc_df[['mean1', 'var1']] = scaler.fit_transform(feedback_df[['mean1', 'var1']].values)
    feedback_sc_df[['mean2', 'var2']] = scaler.transform(feedback_df[['mean2', 'var2']].values)
    feedback_sc_df[['port_returns1', 'port_returns2', 'pareto_fr', 'pareto_fr_w']] = feedback_df[['port_returns1', 'port_returns2', 'pareto_fr', 'pareto_fr_w']].copy()
    joblib.dump(scaler, name)
    return feedback_sc_df

def get_eff_port(eff_front_df, GT_model_scaler_path, GT_model, betas, use_GT=True, plot_eff_front=False):
    scaler = joblib.load(GT_model_scaler_path)
    eff_front_df[['mean_sc', 'var_sc']] = scaler.transform(eff_front_df[['mean', 'var']].values)
    eff_front_df['GT_pref_value'] = eff_front_df[['mean_sc', 'var_sc']].apply(lambda x: GT_model(x), axis=1)
    eff_front_df['Fit_pref_values'] = eff_front_df[['mean_sc', 'var_sc']].apply(lambda x: fit_pref_model(x, betas), axis=1)
    eff_front_df['Fit_pref_values_mean'] = eff_front_df.Fit_pref_values.apply(lambda x: x.mean())

    GT_eff_port = eff_front_df['weights'].iloc[eff_front_df.GT_pref_value.idxmax()]
    Fit_eff_port = eff_front_df['weights'].iloc[eff_front_df.Fit_pref_values_mean.idxmax()]

    if plot_eff_front:
        fig = px.scatter(eff_front_df, x='var', y='mean', color='Fit_pref_values_mean', labels={'Fit_pref_values_mean':'Preference Values Mean'})

        pref_mean, pref_var = eff_front_df[['mean', 'var']].iloc[eff_front_df.GT_pref_value.idxmax()]

        fig.add_trace(go.Scatter(x=[pref_var], y=[pref_mean],
                                mode = 'markers',
                                marker_symbol = 'star',
                                marker_size = 15))

        fig.update_layout(showlegend=False)
        fig.show()
        
    return GT_eff_port if use_GT else Fit_eff_port

def smsemoa_port_opt(daily_returns_df, betas, scaler, plot_eff_front=False):
    
    problem = PortfolioProblem(daily_returns_df)
    algorithm = SMSEMOA(repair=PortfolioRepair())
    res = minimize(problem,
                   algorithm,
                   verbose=False)

    X, F = res.opt.get("X", "F")
    eff_front_df = pd.DataFrame()
    eff_front_df["mean"] = -F[:, 1]
    eff_front_df["var"] = F[:, 0]
    eff_front_df[['mean_sc', 'var_sc']] = scaler.transform(eff_front_df[['mean', 'var']].values)
    F = eff_front_df[['mean_sc', 'var_sc']].values * [1, -1]
    eff_front_df["weights"] = X.tolist()
    pref_model_values = F @ betas.T
    eff_front_df["pref_model_values"] = pref_model_values.tolist()
    eff_front_df["pref_model_mean"] = pref_model_values.mean(axis=1)
    pref_mean, pref_var, eff_port = eff_front_df[['mean', 'var', 'weights']].iloc[eff_front_df.pref_model_mean.idxmax()]
    # eff_front_df["pref_model"] = eff_front_df[["mean", "var"]].apply(lambda x: GT_pref_model(x), axis=1)
    # pref_mean, pref_var, eff_port = eff_front_df[['mean', 'var', 'weights']].iloc[eff_front_df.pref_model.idxmax()]

    if plot_eff_front:
        fig = px.scatter(eff_front_df, x='var', y='mean', color='pref_model_mean')
        fig.add_trace(go.Scatter(x=[pref_var], y=[pref_mean],
                                mode = 'markers',
                                marker_symbol = 'star',
                                marker_size = 15))

        fig.update_layout(showlegend=False)
        fig.show()
    return eff_port

def scipy_port_opt_GT(mean_vec, cov_mat, scaler, GT_pref_model, action_space_shape):
    portfolio_return = lambda x: x @ mean_vec
    portfolio_variance = lambda x: (x.reshape(-1,1).T @ cov_mat @ x.reshape(-1,1)).item()
    preference_function = lambda x: -GT_pref_model(scaler.transform(np.array([[portfolio_return(x), portfolio_variance(x)]])).flatten())
    #preference_function = lambda x: -(betas[:,0]*portfolio_return(x) - betas[:,1]*portfolio_variance(x)).mean()
    #preference_function = lambda x: -(scaler.transform(np.array([[portfolio_return(x), portfolio_variance(x)]])) @ betas.T).flatten().mean()

    constraints = [{"type":"eq", "fun":lambda x: sum(x) - 1}]
    bnds = [(0, None) for i in range(action_space_shape)]
    x = np.ones(action_space_shape)

    res = scipy.optimize.minimize(fun=preference_function, x0=x, constraints=constraints, bounds=bnds)
    return res.x.tolist()

def scipy_port_opt(mean_vec, cov_mat, scaler, betas, action_space_shape):
    portfolio_return = lambda x: x @ mean_vec
    portfolio_variance = lambda x: (x.reshape(-1,1).T @ cov_mat @ x.reshape(-1,1)).item()
    #preference_function = lambda x: -GT_pref_model(scaler.transform(np.array([[portfolio_return(x), portfolio_variance(x)]])).flatten())
    #preference_function = lambda x: -(betas[:,0]*portfolio_return(x) - betas[:,1]*portfolio_variance(x)).mean()
    preference_function = lambda x: -((scaler.transform(np.array([[portfolio_return(x), portfolio_variance(x)]])) * [1, -1]) @ betas.T).flatten().mean()

    constraints = [{"type":"eq", "fun":lambda x: sum(x) - 1}]
    bnds = [(0, None) for i in range(action_space_shape)]
    x = np.ones(action_space_shape)

    res = scipy.optimize.minimize(fun=preference_function, x0=x, constraints=constraints, bounds=bnds)
    return res.x.tolist()

def plot_backtest_results(results_df, mode="Criteria"):
    if mode == "Criteria":
        fig = make_subplots(
            rows=2, cols=3,
            row_heights=[0.5, 0.5],
            subplot_titles=("Portfolio Value", "Criteria: Mean<br>(Daily)",
                            "Criteria: Variance<br>(Daily)", "Preference Values<br>(Daily)"),
            specs=[[{"type": "scatter", "colspan": 3}, None, None],
                [{"type": "histogram"}, {"type": "histogram"}, {"type": "histogram"}]])

        temp_fig = px.line(results_df, x="Date", y="PV", color="Strat")
        for i in range(len(temp_fig["data"])):
            fig.add_trace(go.Scattergl(temp_fig["data"][i]), row=1, col=1)

        features = ["Mean", "Var", "Reward"]
        feature_means = pd.pivot_table(results_df, values=["Mean", "Var", "Reward"],
                                    columns=["Strat"], aggfunc=np.mean)
        for i in range(len(features)):
            temp_fig = px.box(results_df, x="Strat", y=features[i], color="Strat")
            feature = features[i]
            for j in range(len(temp_fig["data"])):
                fig.add_trace(go.Box(temp_fig["data"][j], showlegend=False), row=2, col=i+1)
                strat = temp_fig["data"][j].name
                fig.add_hline(y=feature_means.loc[feature, strat],
                            line_color=temp_fig.data[j].marker.color, row=2, col=i+1)
                fig.update_xaxes(showticklabels=False, row=2, col=i+1)

        fig.update_layout(title_text="Proposed Methods VS Benchmark")
        fig.show()
    elif mode == "Cum_Reward":
        fig = make_subplots(
            rows=2, cols=1,
            row_heights=[0.5, 0.5],
            subplot_titles=("Portfolio Value", "Cumulative Reward"))

        temp_fig_1 = px.line(results_df, x="Date", y="PV", color="Strat")
        temp_fig_2 = px.line(results_df, x="Date", y="Cum_Reward", color="Strat")
        for i in range(len(temp_fig_1["data"])):
            fig.add_trace(go.Scattergl(temp_fig_1["data"][i]), row=1, col=1)
            fig.add_trace(go.Scattergl(temp_fig_2["data"][i], showlegend=False), row=2, col=1)

        fig.update_layout(margin=dict(l=20, r=20, t=20, b=20), template="simple_white", font=dict(family="Times New Roman", size=20),
                          legend=dict(font=dict(family="Times New Roman", size=15)))
        fig.update_annotations(font_size=20)
        fig.show()
    else:
        fig = make_subplots(
            rows=2, cols=1,
            row_heights=[0.5, 0.5],
            subplot_titles=("Portfolio Value", "Reward"))

        temp_fig_1 = px.line(results_df, x="Date", y="PV", color="Strat")
        temp_fig_2 = px.line(results_df, x="Date", y="Reward", color="Strat")
        for i in range(len(temp_fig_1["data"])):
            fig.add_trace(go.Scattergl(temp_fig_1["data"][i]), row=1, col=1)
            fig.add_trace(go.Scattergl(temp_fig_2["data"][i], showlegend=False), row=2, col=1)

        fig.update_layout(margin=dict(l=20, r=20, t=20, b=20), template="simple_white")
        fig.show()

    return fig

def separate_df_regimes(train_df, freq):
    start_date, end_date = train_df.date.unique()[[0,-1]]
    date_range = pd.date_range(start_date, end_date, freq=freq).format()
    date_range[0], date_range[-1] = start_date, end_date
    train_dfs = [data_split(train_df, date_range[0], date_range[1])]
    for i in range(1, len(date_range) - 1):
        train_dfs.append(data_split(train_df, train_dfs[-1].date.unique()[-1], date_range[i + 1]))
    return train_dfs


def create_feedback_dfs_regimes(train_df, num_comparisons, freq, GT_pref_model, regimes_name):
    train_dfs = separate_df_regimes(train_df, freq)
    scaler_path = "./Params/GT_model_data_regimes" + "/" + regimes_name + "/scalers/"
    data_path = "./Data/GT_pref_data_regimes" + "/" + regimes_name + "/"
    for i in range(len(train_dfs)):
        data_file_name = f"GT_model_data_{i}.json"
        scaler_file_name = f"scaler_{i}.joblib"
        feedback_df = create_comparison_dataset(train_dfs[i], num_comparisons)
        feedback_sc_df = scale_feedback_df(feedback_df, scaler_path + scaler_file_name)
        model = GT_pref_model(regime=i, num_regimes=len(train_dfs))
        pref_portfolios = get_preferred_portfolio(feedback_sc_df, model)
        rebalance_classes(feedback_sc_df, 2)
        feedback_sc_df.reset_index().to_json(data_path + data_file_name)
        print(f"Successfully saved {data_file_name}")

def Fit_pref_model_regimes(feedback_sc_dfs):
    regimes_betas_df = pd.DataFrame()
    for i in range(len(feedback_sc_dfs)):
        with pm.Model() as preference_model:
            beta = pm.Dirichlet("beta", a=np.ones(2)/2)
            sigma = pm.Exponential("sigma", 1)
            
            vm = beta[0]*feedback_sc_dfs[i].mean2.values + beta[1]*-feedback_sc_dfs[i].var2.values
            vr = beta[0]*feedback_sc_dfs[i].mean1.values + beta[1]*-feedback_sc_dfs[i].var1.values

            Norm_dist = pm.Normal.dist(0, 1)
            p = 1 - pm.math.exp(Norm_dist.logcdf(-(vm - vr)/pm.math.sqrt(2*sigma**2)))

            Y_obs = pm.Bernoulli("Y_obs", p=p, observed=feedback_sc_dfs[i].pref_portfolio.values)
            trace = pm.sample(1000, tune = 10000, return_inferencedata=False, target_accept=0.999)
            clear_output(wait=True)
        
        regimes_betas_df[f"regime_{i}"] = trace["beta"].tolist()
    return regimes_betas_df

class PortTradingEnv(gym.Env):
    def __init__(self, 
                 df,
                 stock_dim,
                 #hmax,
                 initial_amount,
                 #transaction_cost_pct,
                 #reward_scaling,
                 #state_space,
                 action_space,
                 criteria_weights,
                 tech_indicator_list,
                 scaler,
                 #turbulence_threshold=None,
                 lookback=252,
                 day=0):
        super(PortTradingEnv, self).__init__()
        self.start_day = day
        self.lookback = lookback
        self.df = df
        self.stock_dim = stock_dim
        #self.hmax = hmax
        self.initial_amount = initial_amount
        #self.transaction_cost_pct = transaction_cost_pct
        #self.reward_scaling = reward_scaling
        #self.state_space = state_space
        self.action_space = action_space
        self.tech_indicator_list = tech_indicator_list
        self.criteria_weights = criteria_weights
        
        self.scaler = scaler

        self.action_space = spaces.Box(low=-15, high=15, shape=(self.action_space,))
        # self.observation_space = spaces.Box(low=-np.inf, high=np.inf, 
        #                                     shape=(self.stock_dim + len(self.tech_indicator_list) + 4 + 1,
        #                                            self.stock_dim))
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, 
                                            shape=(self.stock_dim + len(self.tech_indicator_list) + 30 + 1,
                                                   self.stock_dim))

        self.done = False
        self.last_first_moment = 0
        self.last_second_moment = 0
        self.portfolio_value = self.initial_amount
        self.portfolio_value_memory = []
        self.portfolio_return_memory = []
        self.actions_memory=[]
        self.date_memory=[]
        self.reward_memory = []
        self.criteria_memory = []
        self.scaled_criteria_memory = []

    def reset(self):
        self.done = False
        self.last_first_moment = 0
        self.last_second_moment = 0
        self.day = self.start_day
        self.portfolio_value = self.initial_amount
        self.portfolio_value_memory = []
        self.portfolio_return_memory = []
        self.actions_memory=[]
        self.reward = 0
        self.reward_memory = []
        self.criteria_memory = []
        self.scaled_criteria_memory = []
        self.state = self._get_observation()
        return self.state
    
    def step(self, actions, normalize=True, true_model=None):

        self.done = self.day >= len(self.df.index.unique()) - 1

        if self.done:
            return self.state, self.reward, self.done,{}
        else:
            weights = self.softmax_normalization(actions) if normalize else actions
            self.actions_memory.append(weights.tolist())

            #self.date_memory.append(self.data['date'].unique()[0])

            self.day += 1

            self.state = self._get_observation()

            # criteria_values = np.zeros(self.criteria_weights.shape[1])
            # criteria_values[0] = weights @ self.means
            # criteria_values[1] = (weights @ self.covs @ weights)
            # self.criteria_memory.append(criteria_values.tolist())
            # scaled_criteria_values = self.scaler.transform(criteria_values.reshape((1, -1)))[0]
            # scaled_criteria_values[1] = -scaled_criteria_values[1]
            # self.scaled_criteria_memory.append(scaled_criteria_values.tolist())
            # self.reward = self._get_reward(scaled_criteria_values, true_model)
            # self.reward_memory.append(self.reward)

            last_day_returns = self.daily_returns.iloc[-1,:].values
            portfolio_return = weights @ last_day_returns
            self.portfolio_return_memory.append(portfolio_return)
            self.portfolio_value *= 1 + portfolio_return
            self.portfolio_value_memory.append(self.portfolio_value)

            criteria_values = np.zeros(self.criteria_weights.shape[1])
            max_minus_min = self.scaler.data_max_ - self.scaler.data_min_
            criteria_values[0] = (portfolio_return - self.last_first_moment)/max_minus_min[0]
            criteria_values[1] = ((portfolio_return**2 - self.last_second_moment) - 2*(self.last_first_moment + (portfolio_return - self.last_first_moment)/len(self.portfolio_return_memory))*(portfolio_return - self.last_first_moment))/max_minus_min[1]
            self.last_first_moment += (portfolio_return - self.last_first_moment)/len(self.portfolio_return_memory)
            self.last_second_moment += (portfolio_return**2 - self.last_second_moment)/len(self.portfolio_return_memory)
            #criteria_values[0] = np.array(self.portfolio_return_memory).mean()
            #criteria_values[1] = np.array(self.portfolio_return_memory).var()
            self.criteria_memory.append(criteria_values.tolist())
            #scaled_criteria_values = self.scaler.transform(criteria_values.reshape((1, -1)))[0]
            scaled_criteria_values = criteria_values.copy()
            scaled_criteria_values[1] = -scaled_criteria_values[1]
            self.scaled_criteria_memory.append(scaled_criteria_values.tolist())
            self.reward = (1/len(self.portfolio_return_memory))*self._get_reward(scaled_criteria_values, true_model)
            self.reward_memory.append(self.reward)

            return self.state, self.reward, self.done, {}

    def _get_observation(self):
        self.data = self.df.loc[self.day,:]
        self.date_memory.append(self.data['date'].unique()[0])
        self.covs = self.data['cov_list'].values[0]
        self.daily_returns = self.data['return_list'].values[0]
        self.means = self.daily_returns.iloc[-126:].mean().values
        ohlc_list = ['open', 'high', 'low', 'close']
        self.ohlc = [self.data[x].values.tolist() for x in ohlc_list]
        self.tech = [self.data[x].values.tolist() for x in self.tech_indicator_list]
        #return np.vstack((self.covs, self.means, self.ohlc, self.tech))
        return np.vstack((self.covs, self.means, self.tech, self.daily_returns.iloc[-30:].values))
    
    def _get_reward(self, criteria_values, true_model):
        if true_model != None:
            return true_model(criteria_values)
        else:
            preference_values = np.zeros(self.criteria_weights.shape[0])
            for i in range(self.criteria_weights.shape[1]):
                preference_values += self.criteria_weights[:, i] * criteria_values[i] 
            return preference_values.mean()
    
    def softmax_normalization(self, actions):
        numerator = np.exp(actions)
        denominator = np.sum(np.exp(actions))
        softmax_output = numerator/denominator
        return softmax_output
    
    def get_sb_env(self):
        e = DummyVecEnv([lambda: self])
        obs = e.reset()
        return e, obs