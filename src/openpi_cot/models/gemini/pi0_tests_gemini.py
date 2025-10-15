# pi0_test.py

import unittest
import jax
import jax.numpy as jnp
from typing import TYPE_CHECKING
import jax.tree_util as jtu # Import the tree utility library

# Import the components to be tested
from openpi_cot.models.gemini.pi0_config_gemini import Pi0Config

from openpi.models import model as _model

class TestPi0Gemma3(unittest.TestCase):
    """Unit tests for the Gemma 3-based Pi0 model."""

    def setUp(self):
        """Set up common variables for all tests."""
        self.key = jax.random.PRNGKey(0)
        self.batch_size = 2

    def _run_test(self, pi05_enabled: bool):
        """A helper function to run tests for both Pi0 and Pi0.5 configs."""
        # 1. Create the configuration
        config = Pi0Config(
            action_dim=7,
            action_horizon=10,
            max_token_len=32,
            pi05=pi05_enabled
        )

        print(f"Successfully created Pi0Config with pi05={pi05_enabled}")

        # 2a. Get the abstract specs for the dummy data.
        obs_spec, actions_spec = config.inputs_spec(batch_size=self.batch_size)

        # 2b. Create REAL, concrete JAX arrays from the specs.
        # jtu.tree_map walks the nested structure and applies the lambda to each leaf.
        dummy_obs = jtu.tree_map(
            lambda spec: jnp.zeros(spec.shape, spec.dtype), obs_spec
        )
        dummy_actions = jtu.tree_map(
            lambda spec: jnp.zeros(spec.shape, spec.dtype), actions_spec
        )
        print(f"Dummy observation and actions created with batch size {self.batch_size}")
        
        # --- End of Fix ---

        # 3. Create the model instance
        # This tests that initialization works without errors.
        model = config.create(rng=self.key)
        print(f"Successfully created Pi0 model with pi05={pi05_enabled}")
        #self.assertIsInstance(model, Pi0)

        # 4. Test the training forward pass (`compute_loss`)
        loss = model.compute_loss(
            rng=self.key,
            observation=dummy_obs,
            actions=dummy_actions,
            train=True
        )

        print(f"Loss computed with shape {loss.shape} and dtype {loss.dtype}")

        # Check that the loss has the correct shape (batch_size, action_horizon)
        expected_loss_shape = (self.batch_size, config.action_horizon)
        self.assertEqual(loss.shape, expected_loss_shape)
        self.assertTrue(jnp.issubdtype(loss.dtype, jnp.floating))
        print(f"TestPi0(pi05={pi05_enabled}): compute_loss check PASSED.")

        # 5. Test the inference forward pass (`sample_actions`)
        sampled_actions = model.sample_actions(
            rng=self.key,
            observation=dummy_obs,
            num_steps=2 # Use a small number of steps for a fast test
        )

        # Check that the output actions have the same shape as the input actions
        self.assertEqual(sampled_actions.shape, dummy_actions.shape)
        self.assertTrue(jnp.issubdtype(sampled_actions.dtype, jnp.floating))
        print(f"TestPi0(pi05={pi05_enabled}): sample_actions check PASSED.")

    def test_standard_pi0(self):
        """Tests the model with pi05=False (explicit state token)."""
        print("\n--- Testing Standard Pi0 (pi05=False) ---")
        self._run_test(pi05_enabled=False)

    def test_pi05(self):
        """Tests the model with pi05=True (AdaRMSNorm conditioning)."""
        print("\n--- Testing Pi0.5 (pi05=True) ---")
        self._run_test(pi05_enabled=True)


if __name__ == "__main__":
    unittest.main()