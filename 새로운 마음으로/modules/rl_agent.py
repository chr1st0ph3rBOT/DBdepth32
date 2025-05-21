# modules/rl_agent.py

import tensorflow as tf
from tensorflow.keras import layers
import numpy as np

class ReplayBuffer:
    def __init__(self, max_size, state_dim, action_dim):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.state = np.zeros((max_size, state_dim), dtype=np.float32)
        self.action = np.zeros((max_size, action_dim), dtype=np.float32)
        self.reward = np.zeros((max_size, 1), dtype=np.float32)
        self.next_state = np.zeros((max_size, state_dim), dtype=np.float32)
        self.done = np.zeros((max_size, 1), dtype=np.float32)

    def add(self, state, action, reward, next_state, done):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.reward[self.ptr] = reward
        self.next_state[self.ptr] = next_state
        self.done[self.ptr] = done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        idxs = np.random.randint(0, self.size, size=batch_size)
        return dict(
            state=self.state[idxs],
            action=self.action[idxs],
            reward=self.reward[idxs],
            next_state=self.next_state[idxs],
            done=self.done[idxs]
        )

class SACAgent:
    def __init__(self, state_dim=256, action_dim=3, action_bound=1.0, gamma=0.99, tau=0.005, alpha=0.2):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_bound = action_bound
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha

        self.actor = self.build_actor()
        self.critic_1 = self.build_critic()
        self.critic_2 = self.build_critic()
        self.target_critic_1 = self.build_critic()
        self.target_critic_2 = self.build_critic()

        self.target_critic_1.set_weights(self.critic_1.get_weights())
        self.target_critic_2.set_weights(self.critic_2.get_weights())

        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=3e-4)
        self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate=3e-4)

    def build_actor(self):
        inputs = layers.Input(shape=(self.state_dim,))
        x = layers.Dense(256, activation='relu')(inputs)
        x = layers.Dense(256, activation='relu')(x)
        outputs = layers.Dense(self.action_dim, activation='tanh')(x)
        scaled_outputs = layers.Lambda(lambda x: x * self.action_bound)(outputs)
        return tf.keras.Model(inputs, scaled_outputs)

    def build_critic(self):
        state_input = layers.Input(shape=(self.state_dim,))
        action_input = layers.Input(shape=(self.action_dim,))
        x = layers.Concatenate()([state_input, action_input])
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dense(256, activation='relu')(x)
        q_value = layers.Dense(1)(x)
        return tf.keras.Model([state_input, action_input], q_value)

    def select_action(self, state_vector):
        state_vector = np.expand_dims(state_vector, axis=0)  # (1, state_dim)
        action = self.actor(state_vector).numpy()[0]
        return action

    def update(self, batch):
        states = batch['state']
        actions = batch['action']
        rewards = batch['reward']
        next_states = batch['next_state']
        dones = batch['done']

        with tf.GradientTape(persistent=True) as tape:
            next_actions = self.actor(next_states)
            target_q1 = self.target_critic_1([next_states, next_actions])
            target_q2 = self.target_critic_2([next_states, next_actions])
            target_q = tf.minimum(target_q1, target_q2)
            target_value = rewards + self.gamma * (1 - dones) * target_q

            current_q1 = self.critic_1([states, actions])
            current_q2 = self.critic_2([states, actions])
            critic_loss = tf.reduce_mean((current_q1 - target_value) ** 2 + (current_q2 - target_value) ** 2)

            new_actions = self.actor(states)
            q1 = self.critic_1([states, new_actions])
            actor_loss = -tf.reduce_mean(q1)

        critic_grads = tape.gradient(critic_loss, self.critic_1.trainable_variables + self.critic_2.trainable_variables)
        self.critic_optimizer.apply_gradients(zip(critic_grads, self.critic_1.trainable_variables + self.critic_2.trainable_variables))

        actor_grads = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))

        self.update_target_network(self.target_critic_1, self.critic_1)
        self.update_target_network(self.target_critic_2, self.critic_2)

    def update_target_network(self, target_model, source_model):
        for target_param, source_param in zip(target_model.trainable_variables, source_model.trainable_variables):
            target_param.assign(self.tau * source_param + (1 - self.tau) * target_param)
