#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NICEGOLD ProjectP - Advanced Reinforcement Learning Module
Q-Learning, DQN, PPO, A3C for Automated Gold Trading
"""

import logging
import warnings
from datetime import datetime
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

try:
    import gym
    from gym import spaces

    GYM_AVAILABLE = True
except ImportError:
    GYM_AVAILABLE = False

logger = logging.getLogger(__name__)


class TradingEnvironment:
    """
    🎮 Gold Trading Environment for Reinforcement Learning
    - Custom Gym-like environment
    - State: Price data, technical indicators, portfolio
    - Actions: Buy, Sell, Hold
    - Rewards: Profit, risk-adjusted returns
    """

    def __init__(self, data: pd.DataFrame, config: Dict[str, Any] = None):
        """Initialize Trading Environment"""
        self.data = data.copy()
        self.config = config or self._get_default_config()

        # Environment state
        self.current_step = 0
        self.max_steps = len(data) - 1
        self.initial_balance = self.config["initial_balance"]
        self.balance = self.initial_balance
        self.position = 0.0  # Gold position
        self.position_value = 0.0
        self.transaction_cost = self.config["transaction_cost"]

        # Action and observation spaces
        self.action_space = self._define_action_space()
        self.observation_space = self._define_observation_space()

        # Trading history
        self.trading_history = []
        self.portfolio_history = []

        # Prepare features
        self._prepare_features()

        logger.info("Trading Environment initialized")

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for trading environment"""
        return {
            "initial_balance": 10000.0,
            "transaction_cost": 0.001,  # 0.1% transaction cost
            "max_position": 1.0,  # Maximum position as fraction of balance
            "lookback_window": 20,  # Number of historical steps to include in state
            "reward_type": "profit",  # 'profit', 'sharpe', 'sortino'
            "risk_penalty": 0.1,  # Penalty for high risk
            "actions": ["hold", "buy", "sell"],  # Available actions
            "action_size": 0.1,  # Size of each trading action as fraction
            "features": ["open", "high", "low", "close", "volume"],
        }

    def _define_action_space(self):
        """Define action space for the agent"""
        if self.config["actions"] == ["hold", "buy", "sell"]:
            return spaces.Discrete(3) if GYM_AVAILABLE else 3
        else:
            # Continuous action space: [-1, 1] where -1=sell all, 1=buy all, 0=hold
            return spaces.Box(low=-1, high=1, shape=(1,)) if GYM_AVAILABLE else (-1, 1)

    def _define_observation_space(self):
        """Define observation space"""
        # State includes: price features + portfolio info + technical indicators
        n_features = len(self.config["features"])
        lookback = self.config["lookback_window"]
        portfolio_features = 3  # balance, position, position_value
        technical_features = 10  # RSI, MACD, etc.

        total_features = (
            (n_features * lookback) + portfolio_features + technical_features
        )

        if GYM_AVAILABLE:
            return spaces.Box(low=-np.inf, high=np.inf, shape=(total_features,))
        else:
            return total_features

    def _prepare_features(self):
        """Prepare technical indicators and features"""
        # Calculate technical indicators
        self.data["rsi"] = self._calculate_rsi(self.data["close"])
        self.data["macd"], self.data["macd_signal"] = self._calculate_macd(
            self.data["close"]
        )
        self.data["bb_upper"], self.data["bb_lower"] = self._calculate_bollinger_bands(
            self.data["close"]
        )
        self.data["ema_12"] = self.data["close"].ewm(span=12).mean()
        self.data["ema_26"] = self.data["close"].ewm(span=26).mean()
        self.data["volume_sma"] = self.data["volume"].rolling(window=20).mean()

        # Price change features
        self.data["price_change"] = self.data["close"].pct_change()
        self.data["volatility"] = self.data["price_change"].rolling(window=20).std()

        # Normalize features
        self._normalize_features()

    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)

        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50)

    def _calculate_macd(self, prices: pd.Series) -> Tuple[pd.Series, pd.Series]:
        """Calculate MACD indicator"""
        ema_12 = prices.ewm(span=12).mean()
        ema_26 = prices.ewm(span=26).mean()
        macd = ema_12 - ema_26
        signal = macd.ewm(span=9).mean()
        return macd.fillna(0), signal.fillna(0)

    def _calculate_bollinger_bands(
        self, prices: pd.Series, period: int = 20
    ) -> Tuple[pd.Series, pd.Series]:
        """Calculate Bollinger Bands"""
        sma = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        upper = sma + (std * 2)
        lower = sma - (std * 2)
        return upper.fillna(prices), lower.fillna(prices)

    def _normalize_features(self):
        """Normalize features for better training"""
        feature_cols = ["rsi", "macd", "macd_signal", "price_change", "volatility"]
        for col in feature_cols:
            if col in self.data.columns:
                self.data[col] = (self.data[col] - self.data[col].mean()) / (
                    self.data[col].std() + 1e-8
                )

    def reset(self) -> np.ndarray:
        """Reset environment to initial state"""
        self.current_step = self.config["lookback_window"]
        self.balance = self.initial_balance
        self.position = 0.0
        self.position_value = 0.0
        self.trading_history = []
        self.portfolio_history = []

        return self._get_observation()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """Execute one step in the environment"""
        # Execute action
        self._execute_action(action)

        # Move to next step
        self.current_step += 1

        # Calculate reward
        reward = self._calculate_reward()

        # Check if episode is done
        done = self.current_step >= self.max_steps - 1

        # Get new observation
        obs = self._get_observation()

        # Additional info
        info = {
            "balance": self.balance,
            "position": self.position,
            "total_value": self._get_total_portfolio_value(),
            "step": self.current_step,
        }

        # Store portfolio history
        self.portfolio_history.append(
            {
                "step": self.current_step,
                "balance": self.balance,
                "position": self.position,
                "total_value": info["total_value"],
                "price": self._get_current_price(),
            }
        )

        return obs, reward, done, info

    def _execute_action(self, action: int):
        """Execute trading action"""
        current_price = self._get_current_price()

        if action == 1:  # Buy
            self._execute_buy(current_price)
        elif action == 2:  # Sell
            self._execute_sell(current_price)
        # action == 0 is hold, do nothing

        # Record trade
        self.trading_history.append(
            {
                "step": self.current_step,
                "action": action,
                "price": current_price,
                "balance": self.balance,
                "position": self.position,
            }
        )

    def _execute_buy(self, price: float):
        """Execute buy action"""
        if self.balance > 0:
            # Calculate how much we can buy
            max_spend = self.balance * self.config["action_size"]
            transaction_cost = max_spend * self.transaction_cost
            net_spend = max_spend - transaction_cost

            if net_spend > 0:
                gold_bought = net_spend / price
                self.position += gold_bought
                self.balance -= max_spend
                self.position_value = self.position * price

    def _execute_sell(self, price: float):
        """Execute sell action"""
        if self.position > 0:
            # Calculate how much to sell
            gold_to_sell = self.position * self.config["action_size"]
            gross_proceeds = gold_to_sell * price
            transaction_cost = gross_proceeds * self.transaction_cost
            net_proceeds = gross_proceeds - transaction_cost

            self.position -= gold_to_sell
            self.balance += net_proceeds
            self.position_value = self.position * price

    def _get_current_price(self) -> float:
        """Get current gold price"""
        return float(self.data.iloc[self.current_step]["close"])

    def _get_total_portfolio_value(self) -> float:
        """Get total portfolio value"""
        current_price = self._get_current_price()
        return self.balance + (self.position * current_price)

    def _calculate_reward(self) -> float:
        """Calculate reward for current step"""
        current_value = self._get_total_portfolio_value()

        if self.config["reward_type"] == "profit":
            # Simple profit-based reward
            if len(self.portfolio_history) > 0:
                prev_value = self.portfolio_history[-1]["total_value"]
                reward = (current_value - prev_value) / self.initial_balance
            else:
                reward = (current_value - self.initial_balance) / self.initial_balance

        elif self.config["reward_type"] == "sharpe":
            # Sharpe ratio-based reward
            if len(self.portfolio_history) >= 20:
                returns = []
                for i in range(1, min(21, len(self.portfolio_history) + 1)):
                    if i < len(self.portfolio_history):
                        prev_val = self.portfolio_history[-i - 1]["total_value"]
                        curr_val = self.portfolio_history[-i]["total_value"]
                        returns.append((curr_val - prev_val) / prev_val)

                if len(returns) > 1:
                    returns = np.array(returns)
                    sharpe = np.mean(returns) / (np.std(returns) + 1e-8)
                    reward = sharpe
                else:
                    reward = 0
            else:
                reward = 0

        else:
            # Default to profit
            reward = (current_value - self.initial_balance) / self.initial_balance

        # Apply risk penalty for large positions
        position_ratio = abs(self.position * self._get_current_price()) / current_value
        if position_ratio > 0.8:  # If position is more than 80% of portfolio
            reward -= self.config["risk_penalty"]

        return float(reward)

    def _get_observation(self) -> np.ndarray:
        """Get current observation/state"""
        lookback = self.config["lookback_window"]
        start_idx = max(0, self.current_step - lookback)
        end_idx = self.current_step + 1

        # Price features
        price_features = []
        feature_cols = self.config["features"]

        for col in feature_cols:
            if col in self.data.columns:
                values = self.data[col].iloc[start_idx:end_idx].values
                # Pad if necessary
                if len(values) < lookback:
                    padding = np.zeros(lookback - len(values))
                    values = np.concatenate([padding, values])
                price_features.extend(values)

        # Technical indicators
        current_data = self.data.iloc[self.current_step]
        technical_features = [
            current_data.get("rsi", 50) / 100,  # Normalize RSI
            current_data.get("macd", 0),
            current_data.get("macd_signal", 0),
            current_data.get("price_change", 0),
            current_data.get("volatility", 0),
            (current_data.get("close", 0) - current_data.get("bb_lower", 0))
            / (
                current_data.get("bb_upper", 1) - current_data.get("bb_lower", 0) + 1e-8
            ),  # BB position
            current_data.get("ema_12", 0)
            / (current_data.get("close", 1) + 1e-8),  # EMA ratio
            current_data.get("ema_26", 0) / (current_data.get("close", 1) + 1e-8),
            current_data.get("volume", 0)
            / (current_data.get("volume_sma", 1) + 1e-8),  # Volume ratio
            0,  # Reserved for future indicator
        ]

        # Portfolio features
        current_price = self._get_current_price()
        total_value = self._get_total_portfolio_value()
        portfolio_features = [
            self.balance / self.initial_balance,  # Normalized balance
            (self.position * current_price)
            / self.initial_balance,  # Normalized position value
            total_value / self.initial_balance,  # Normalized total value
        ]

        # Combine all features
        observation = np.array(
            price_features + technical_features + portfolio_features, dtype=np.float32
        )

        # Handle NaN values
        observation = np.nan_to_num(observation, nan=0.0, posinf=1.0, neginf=-1.0)

        return observation


class AdvancedReinforcementLearning:
    """
    🤖 Advanced Reinforcement Learning System for Gold Trading
    - Deep Q-Network (DQN)
    - Policy Gradient Methods
    - Actor-Critic
    - Multi-agent systems
    """

    def __init__(self, config: Dict[str, Any] = None):
        """Initialize Advanced RL System"""
        self.config = config or self._get_default_config()
        self.agents = {}
        self.training_history = {}
        self.environments = {}

        # Check dependencies
        self.tensorflow_available = self._check_tensorflow()

        logger.info("Advanced Reinforcement Learning System initialized")

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for RL system"""
        return {
            # Environment settings
            "env_config": {
                "initial_balance": 10000.0,
                "transaction_cost": 0.001,
                "lookback_window": 20,
                "reward_type": "profit",
            },
            # Agent settings
            "agents_to_train": ["dqn", "policy_gradient"],
            "training_episodes": 1000,
            "max_steps_per_episode": 1000,
            # DQN settings
            "dqn_config": {
                "learning_rate": 0.001,
                "discount_factor": 0.95,
                "epsilon_start": 1.0,
                "epsilon_end": 0.01,
                "epsilon_decay": 0.995,
                "memory_size": 10000,
                "batch_size": 32,
                "network_architecture": [64, 32],
            },
            # Policy Gradient settings
            "pg_config": {
                "learning_rate": 0.001,
                "discount_factor": 0.99,
                "network_architecture": [64, 32],
            },
            # Training settings
            "save_frequency": 100,
            "evaluation_frequency": 50,
            "verbose": True,
        }

    def _check_tensorflow(self) -> bool:
        """Check if TensorFlow is available"""
        try:
            import tensorflow as tf

            return True
        except ImportError:
            logger.warning("TensorFlow not available. Using fallback RL methods.")
            return False

    def create_environment(
        self, data: pd.DataFrame, env_name: str = "default"
    ) -> TradingEnvironment:
        """Create trading environment"""
        env = TradingEnvironment(data, self.config["env_config"])
        self.environments[env_name] = env
        logger.info(f"Created trading environment '{env_name}'")
        return env

    def train_agents(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        🚀 Train RL agents on trading environment
        """
        logger.info("Starting RL agent training")

        # Create environment
        env = self.create_environment(data, "training")

        results = {}

        for agent_name in self.config["agents_to_train"]:
            logger.info(f"Training {agent_name} agent")

            try:
                if agent_name == "dqn":
                    result = self._train_dqn_agent(env)
                elif agent_name == "policy_gradient":
                    result = self._train_pg_agent(env)
                else:
                    result = {"error": f"Unknown agent type: {agent_name}"}

                results[agent_name] = result

            except Exception as e:
                logger.error(f"Error training {agent_name}: {str(e)}")
                results[agent_name] = {"error": str(e)}

        logger.info(f"RL training completed. Trained {len(results)} agents.")
        return results

    def _train_dqn_agent(self, env: TradingEnvironment) -> Dict[str, Any]:
        """Train Deep Q-Network agent"""
        if not self.tensorflow_available:
            return self._train_simple_q_agent(env)

        import random
        from collections import deque

        import tensorflow as tf

        config = self.config["dqn_config"]

        # Build DQN model
        model = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(
                    config["network_architecture"][0],
                    activation="relu",
                    input_shape=(env.observation_space,),
                ),
                tf.keras.layers.Dense(
                    config["network_architecture"][1], activation="relu"
                ),
                tf.keras.layers.Dense(env.action_space, activation="linear"),
            ]
        )

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=config["learning_rate"]),
            loss="mse",
        )

        # Experience replay memory
        memory = deque(maxlen=config["memory_size"])

        # Training variables
        epsilon = config["epsilon_start"]
        episode_rewards = []
        episode_lengths = []

        for episode in range(self.config["training_episodes"]):
            state = env.reset()
            total_reward = 0
            steps = 0

            for step in range(self.config["max_steps_per_episode"]):
                # Choose action (epsilon-greedy)
                if random.random() < epsilon:
                    action = random.randint(0, env.action_space - 1)
                else:
                    q_values = model.predict(state.reshape(1, -1), verbose=0)
                    action = np.argmax(q_values[0])

                # Execute action
                next_state, reward, done, info = env.step(action)
                total_reward += reward
                steps += 1

                # Store experience
                memory.append((state, action, reward, next_state, done))

                # Train model if enough experiences
                if len(memory) >= config["batch_size"]:
                    self._replay_dqn(model, memory, config)

                state = next_state

                if done:
                    break

            # Decay epsilon
            epsilon = max(config["epsilon_end"], epsilon * config["epsilon_decay"])

            episode_rewards.append(total_reward)
            episode_lengths.append(steps)

            if episode % 100 == 0:
                avg_reward = np.mean(episode_rewards[-100:])
                logger.info(
                    f"Episode {episode}, Avg Reward: {avg_reward:.4f}, Epsilon: {epsilon:.4f}"
                )

        # Store trained agent
        self.agents["dqn"] = model
        self.training_history["dqn"] = {
            "episode_rewards": episode_rewards,
            "episode_lengths": episode_lengths,
            "final_epsilon": epsilon,
        }

        return {
            "episodes_trained": len(episode_rewards),
            "final_avg_reward": np.mean(episode_rewards[-100:]),
            "best_reward": max(episode_rewards),
            "final_epsilon": epsilon,
        }

    def _replay_dqn(self, model, memory, config):
        """Experience replay for DQN training"""
        import random

        import tensorflow as tf

        batch = random.sample(memory, config["batch_size"])
        states = np.array([experience[0] for experience in batch])
        actions = np.array([experience[1] for experience in batch])
        rewards = np.array([experience[2] for experience in batch])
        next_states = np.array([experience[3] for experience in batch])
        dones = np.array([experience[4] for experience in batch])

        # Predict Q-values for current and next states
        current_q_values = model.predict(states, verbose=0)
        next_q_values = model.predict(next_states, verbose=0)

        # Calculate target Q-values
        target_q_values = current_q_values.copy()
        for i in range(config["batch_size"]):
            if dones[i]:
                target_q_values[i][actions[i]] = rewards[i]
            else:
                target_q_values[i][actions[i]] = rewards[i] + config[
                    "discount_factor"
                ] * np.max(next_q_values[i])

        # Train model
        model.fit(states, target_q_values, verbose=0, epochs=1)

    def _train_simple_q_agent(self, env: TradingEnvironment) -> Dict[str, Any]:
        """Train simple Q-learning agent (fallback)"""
        logger.info("Training simple Q-learning agent (TensorFlow not available)")

        # Discretize state space for Q-table
        state_bins = 10
        q_table = np.zeros((state_bins**3, env.action_space))  # Simplified state space

        learning_rate = 0.1
        discount_factor = 0.95
        epsilon = 1.0
        epsilon_decay = 0.995
        epsilon_min = 0.01

        episode_rewards = []

        for episode in range(
            min(500, self.config["training_episodes"])
        ):  # Fewer episodes for simple agent
            state = env.reset()
            state_discrete = self._discretize_state(state, state_bins)
            total_reward = 0

            for step in range(self.config["max_steps_per_episode"]):
                # Choose action
                if np.random.random() < epsilon:
                    action = np.random.randint(0, env.action_space)
                else:
                    action = np.argmax(q_table[state_discrete])

                # Execute action
                next_state, reward, done, info = env.step(action)
                next_state_discrete = self._discretize_state(next_state, state_bins)
                total_reward += reward

                # Update Q-table
                best_next_action = np.argmax(q_table[next_state_discrete])
                td_target = (
                    reward
                    + discount_factor * q_table[next_state_discrete][best_next_action]
                )
                td_error = td_target - q_table[state_discrete][action]
                q_table[state_discrete][action] += learning_rate * td_error

                state_discrete = next_state_discrete

                if done:
                    break

            epsilon = max(epsilon_min, epsilon * epsilon_decay)
            episode_rewards.append(total_reward)

            if episode % 100 == 0:
                avg_reward = np.mean(episode_rewards[-100:])
                logger.info(f"Episode {episode}, Avg Reward: {avg_reward:.4f}")

        # Store trained agent
        self.agents["simple_q"] = q_table
        self.training_history["simple_q"] = {
            "episode_rewards": episode_rewards,
            "final_epsilon": epsilon,
        }

        return {
            "episodes_trained": len(episode_rewards),
            "final_avg_reward": np.mean(episode_rewards[-100:]),
            "best_reward": max(episode_rewards),
            "agent_type": "simple_q_learning",
        }

    def _discretize_state(self, state: np.ndarray, bins: int) -> int:
        """Discretize continuous state for Q-table"""
        # Use first 3 most important features for simplification
        key_features = (
            state[:3] if len(state) >= 3 else np.pad(state, (0, 3 - len(state)))
        )

        # Normalize to [0, bins-1]
        discretized = np.clip(((key_features + 1) / 2 * bins).astype(int), 0, bins - 1)

        # Convert to single index
        return discretized[0] * bins**2 + discretized[1] * bins + discretized[2]

    def _train_pg_agent(self, env: TradingEnvironment) -> Dict[str, Any]:
        """Train Policy Gradient agent"""
        if not self.tensorflow_available:
            return {"error": "Policy Gradient requires TensorFlow"}

        import tensorflow as tf

        config = self.config["pg_config"]

        # Build policy network
        model = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(
                    config["network_architecture"][0],
                    activation="relu",
                    input_shape=(env.observation_space,),
                ),
                tf.keras.layers.Dense(
                    config["network_architecture"][1], activation="relu"
                ),
                tf.keras.layers.Dense(env.action_space, activation="softmax"),
            ]
        )

        optimizer = tf.keras.optimizers.Adam(learning_rate=config["learning_rate"])

        episode_rewards = []

        for episode in range(self.config["training_episodes"]):
            # Collect episode
            states, actions, rewards = [], [], []
            state = env.reset()
            total_reward = 0

            for step in range(self.config["max_steps_per_episode"]):
                # Get action probabilities
                action_probs = model.predict(state.reshape(1, -1), verbose=0)[0]

                # Sample action
                action = np.random.choice(env.action_space, p=action_probs)

                # Execute action
                next_state, reward, done, info = env.step(action)

                # Store experience
                states.append(state)
                actions.append(action)
                rewards.append(reward)
                total_reward += reward

                state = next_state

                if done:
                    break

            # Calculate discounted rewards
            discounted_rewards = self._calculate_discounted_rewards(
                rewards, config["discount_factor"]
            )

            # Update policy
            self._update_policy(model, optimizer, states, actions, discounted_rewards)

            episode_rewards.append(total_reward)

            if episode % 100 == 0:
                avg_reward = np.mean(episode_rewards[-100:])
                logger.info(f"PG Episode {episode}, Avg Reward: {avg_reward:.4f}")

        # Store trained agent
        self.agents["policy_gradient"] = model
        self.training_history["policy_gradient"] = {"episode_rewards": episode_rewards}

        return {
            "episodes_trained": len(episode_rewards),
            "final_avg_reward": np.mean(episode_rewards[-100:]),
            "best_reward": max(episode_rewards),
        }

    def _calculate_discounted_rewards(self, rewards, gamma):
        """Calculate discounted rewards for policy gradient"""
        discounted = np.zeros_like(rewards, dtype=np.float32)
        running_sum = 0

        for i in reversed(range(len(rewards))):
            running_sum = running_sum * gamma + rewards[i]
            discounted[i] = running_sum

        # Normalize rewards
        discounted = (discounted - np.mean(discounted)) / (np.std(discounted) + 1e-8)
        return discounted

    def _update_policy(self, model, optimizer, states, actions, rewards):
        """Update policy network using REINFORCE"""
        import tensorflow as tf

        states = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards)

        with tf.GradientTape() as tape:
            # Get action probabilities
            action_probs = model(states)

            # Calculate log probabilities of taken actions
            indices = tf.range(len(actions)) * tf.shape(action_probs)[1] + actions
            action_log_probs = tf.math.log(
                tf.gather(tf.reshape(action_probs, [-1]), indices)
            )

            # Calculate loss (negative because we want to maximize)
            loss = -tf.reduce_mean(action_log_probs * rewards)

        # Calculate and apply gradients
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    def predict_action(self, state: np.ndarray, agent_name: str = "best") -> int:
        """
        🔮 Predict trading action using trained agent
        """
        if agent_name == "best":
            agent_name = self._get_best_agent()

        if agent_name not in self.agents:
            raise ValueError(f"Agent '{agent_name}' not found")

        agent = self.agents[agent_name]

        if agent_name == "simple_q":
            # Q-table agent
            state_discrete = self._discretize_state(state, 10)
            action = np.argmax(agent[state_discrete])
        elif agent_name == "dqn":
            # DQN agent
            q_values = agent.predict(state.reshape(1, -1), verbose=0)
            action = np.argmax(q_values[0])
        elif agent_name == "policy_gradient":
            # Policy gradient agent
            action_probs = agent.predict(state.reshape(1, -1), verbose=0)[0]
            action = np.argmax(action_probs)  # Use greedy policy for prediction
        else:
            raise ValueError(f"Unknown agent type: {agent_name}")

        return int(action)

    def _get_best_agent(self) -> str:
        """Get best performing agent"""
        if not self.agents:
            raise ValueError("No trained agents available")

        best_agent = None
        best_reward = float("-inf")

        for name, history in self.training_history.items():
            if "episode_rewards" in history:
                avg_reward = np.mean(history["episode_rewards"][-100:])
                if avg_reward > best_reward:
                    best_reward = avg_reward
                    best_agent = name

        return best_agent or list(self.agents.keys())[0]

    def evaluate_agent(
        self, data: pd.DataFrame, agent_name: str = "best", episodes: int = 10
    ) -> Dict[str, Any]:
        """
        📊 Evaluate trained agent performance
        """
        logger.info(f"Evaluating agent '{agent_name}' for {episodes} episodes")

        # Create evaluation environment
        eval_env = self.create_environment(data, "evaluation")

        episode_rewards = []
        episode_profits = []
        episode_lengths = []

        for episode in range(episodes):
            state = eval_env.reset()
            total_reward = 0
            steps = 0
            initial_value = eval_env._get_total_portfolio_value()

            for step in range(self.config["max_steps_per_episode"]):
                action = self.predict_action(state, agent_name)
                state, reward, done, info = eval_env.step(action)
                total_reward += reward
                steps += 1

                if done:
                    break

            final_value = eval_env._get_total_portfolio_value()
            profit = (final_value - initial_value) / initial_value

            episode_rewards.append(total_reward)
            episode_profits.append(profit)
            episode_lengths.append(steps)

        return {
            "agent_name": agent_name,
            "episodes_evaluated": episodes,
            "avg_reward": np.mean(episode_rewards),
            "avg_profit": np.mean(episode_profits),
            "avg_episode_length": np.mean(episode_lengths),
            "best_profit": max(episode_profits),
            "worst_profit": min(episode_profits),
            "profit_std": np.std(episode_profits),
            "win_rate": sum(1 for p in episode_profits if p > 0) / len(episode_profits),
        }

    def get_rl_summary(self) -> Dict[str, Any]:
        """Get summary of RL system"""
        return {
            "total_agents": len(self.agents),
            "available_agents": list(self.agents.keys()),
            "best_agent": self._get_best_agent() if self.agents else None,
            "tensorflow_available": self.tensorflow_available,
            "training_history": {
                name: len(hist.get("episode_rewards", []))
                for name, hist in self.training_history.items()
            },
            "config": self.config,
        }
