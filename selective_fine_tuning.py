# The class in this script implements a selective fine-tuning method based on the condition number
# Author: Oswaldo Ludwig (now with AI support)
# Date: 03/07/2025
# In case of publication using this script or ideas in this script, cite:
# Ludwig, Oswaldo. "The Condition Number as a Scale-Invariant Proxy for Information Encoding in Neural Units." arXiv preprint arXiv:2506.16289 (2025).

import torch
import torch.nn as nn
import torch.optim as optim
import os
import json
import numpy as np
import logging
from typing import Type, Dict, Any, Set, List

# Configure logging (ensure this is at the top level or configured once)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SelectiveFineTuningOptimizer:
    """
    A custom optimizer wrapper that selectively fine-tunes a PyTorch model
    based on the condition numbers of its parameters. Parameters with lower
    condition numbers are prioritized for fine-tuning.
    """
    def __init__(self, model: nn.Module, base_optimizer_cls: Type[optim.Optimizer], optimizer_args: Dict[str, Any],
                 condition_file: str = 'condition_numbers.json',
                 num_tensors_to_finetune: int = 100,
                 recompute: bool = False,
                 max_dim_size_to_analyze: int = None): # New parameter for filtering
        """
        Initializes the SelectiveFineTuningOptimizer.

        Args:
            model (nn.Module): The PyTorch model to be fine-tuned.
            base_optimizer_cls (Type[optim.Optimizer]): The class of the base optimizer (e.g., torch.optim.Adam).
            optimizer_args (Dict[str, Any]): A dictionary of arguments to pass to the base optimizer constructor.
            condition_file (str): Path to the JSON file for storing/loading condition numbers.
            num_tensors_to_finetune (int): The number of top tensors (based on condition number) to fine-tune.
            recompute (bool): If True, recompute condition numbers even if the file exists.
            max_dim_size_to_analyze (int, optional): If provided, any parameter tensor with at least one dimension
                                                    larger than this value will be skipped from analysis.
                                                    Useful for ignoring very large embedding matrices etc.
        """
        self.model = model
        self.condition_file = condition_file
        self.num_tensors_to_finetune = num_tensors_to_finetune
        self.recompute = recompute
        self.max_dim_size_to_analyze = max_dim_size_to_analyze # Store the new parameter

        self.condition_numbers: Dict[str, float] = {}

        if not os.path.exists(condition_file) or recompute:
            self.condition_numbers = self._analyze_model()
            self._save_condition_numbers()
        else:
            self.condition_numbers = self._load_condition_numbers()

        self.trainable_param_names: Set[str] = self._select_trainable_parameters()
        self._unfreeze_selected_parameters()

        # Initialize the base optimizer with selected parameters
        params_to_optimize = [p for n, p in model.named_parameters() if n in self.trainable_param_names]
        if not params_to_optimize:
            logger.warning("No parameters selected for fine-tuning based on the criteria. Optimizer will have no parameters.")
        self.optimizer = base_optimizer_cls(params_to_optimize, **optimizer_args)
        logger.info(f"Optimizer initialized with {len(params_to_optimize)} trainable parameters.")


    def _analyze_model(self) -> Dict[str, float]:
        """
        Analyzes the singular values of model parameters to compute their condition numbers.
        Parameters with less than 2 dimensions or having any dimension
        larger than `max_dim_size_to_analyze` are ignored.
        SVD is performed on the GPU if the tensor is on CUDA, otherwise on CPU.

        Returns:
            Dict[str, float]: A dictionary mapping parameter names to their condition numbers.
        """
        condition_numbers = {}
        logger.info("Analyzing the model tensors...")

        initial_requires_grad_state = {}
        for name, param in self.model.named_parameters():
            initial_requires_grad_state[name] = param.requires_grad
            param.requires_grad = False # Temporarily disable for analysis

        analyzed_count = 0
        skipped_ndim_count = 0
        skipped_dim_size_count = 0 # New counter
        skipped_svd_error_count = 0
        total_params_in_model = 0

        try:
            for name, param in self.model.named_parameters():
                total_params_in_model += 1
                # Filter 1: Skip by number of dimensions
                if param.ndim < 2:
                    logger.debug(f"Skipping {name} due to less than 2 dimensions (ndim={param.ndim}).")
                    skipped_ndim_count += 1
                    continue
                # Filter 2: Skip by any dimension size exceeding threshold
                if self.max_dim_size_to_analyze is not None:
                    if any(dim_size > self.max_dim_size_to_analyze for dim_size in param.shape):
                        logger.debug(f"Skipping {name} due to a dimension larger than {self.max_dim_size_to_analyze} (shape={param.shape}).")
                        skipped_dim_size_count += 1
                        continue

                try:
                    data = param.detach() # Keep on GPU if already there
                    if data.is_cuda:
                        # Perform SVD on GPU
                        u, s, v = torch.linalg.svd(data, full_matrices=False)
                    else:
                        # Fallback to CPU if not on CUDA
                        u, s, v = torch.linalg.svd(data.cpu(), full_matrices=False)

                    cond_number = (s[0] / s[-1]).item() if s[-1] > 0 else float('inf')
                    condition_numbers[name] = cond_number
                    analyzed_count += 1
                    logger.debug(f"Analyzed {name}: condition_number={cond_number:.4f}")
                except torch.linalg.LinAlgError as e:
                    logger.warning(f"Skipping {name} due to SVD Linear Algebra error: {e}")
                    skipped_svd_error_count += 1
                except Exception as e:
                    logger.error(f"Skipping {name} due to unexpected error during SVD: {e}")
                    skipped_svd_error_count += 1
        finally:
            # Restore initial requires_grad state (though _unfreeze_selected_parameters will override this)
            for name, param in self.model.named_parameters():
                param.requires_grad = initial_requires_grad_state[name]


        logger.info(f"Done analyzing model tensors. Total parameters in model: {total_params_in_model}")
        logger.info(f"Parameters analyzed for condition numbers: {analyzed_count}")
        logger.info(f"Skipped due to ndim < 2: {skipped_ndim_count}")
        logger.info(f"Skipped due to dimension size > {self.max_dim_size_to_analyze}: {skipped_dim_size_count}") # New log
        logger.info(f"Skipped due to SVD errors: {skipped_svd_error_count}")
        return condition_numbers

    def _save_condition_numbers(self):
        """
        Saves the computed condition numbers to a JSON file.
        """
        try:
            with open(self.condition_file, 'w') as f:
                json.dump(self.condition_numbers, f, indent=2)
            logger.info(f"Condition numbers saved to {self.condition_file}")
        except IOError as e:
            logger.error(f"Failed to save condition numbers to {self.condition_file}: {e}")

    def _load_condition_numbers(self) -> Dict[str, float]:
        """
        Loads condition numbers from a JSON file. If the file is corrupted,
        it triggers a recomputation.

        Returns:
            Dict[str, float]: The loaded condition numbers.
        """
        try:
            with open(self.condition_file, 'r') as f:
                data = json.load(f)
            logger.info(f"Condition numbers loaded from {self.condition_file}")
            return data
        except json.JSONDecodeError as e:
            logger.warning(f"Condition file '{self.condition_file}' is corrupted or invalid. Error: {e}. Recomputing.")
            if os.path.exists(self.condition_file):
                try:
                    os.remove(self.condition_file) # Remove corrupted file
                    logger.info(f"Removed corrupted condition file: {self.condition_file}")
                except OSError as err:
                    logger.error(f"Error removing corrupted file {self.condition_file}: {err}")
            return self._analyze_model() # Recompute if loading fails
        except IOError as e:
            logger.error(f"Failed to load condition numbers from {self.condition_file}: {e}. Recomputing.")
            return self._analyze_model() # Recompute if file not found or other IO error

    def _select_trainable_parameters(self) -> Set[str]:
        """
        Selects the top `num_tensors_to_finetune` parameters based on their condition numbers
        (lower condition number is better).

        Returns:
            Set[str]: A set of names of the parameters chosen for fine-tuning.
        """
        if not self.condition_numbers:
            logger.warning("No condition numbers available to select trainable parameters.")
            return set()

        sorted_params = sorted(self.condition_numbers.items(), key=lambda x: x[1])
        selected = [name for name, _ in sorted_params[:self.num_tensors_to_finetune]]
        logger.info(f"Selected {len(selected)} parameters for fine-tuning out of {len(self.condition_numbers)} analyzed.")
        logger.debug(f"Selected parameters: {selected}")
        return set(selected)

    def _unfreeze_selected_parameters(self):
        """
        Sets `requires_grad=True` for the selected trainable parameters
        and `requires_grad=False` for all other parameters in the model.
        """
        total_params = 0
        frozen_params_count = 0
        unfrozen_params_count = 0

        for name, param in self.model.named_parameters():
            total_params += 1
            if name in self.trainable_param_names:
                if not param.requires_grad: # Only change if it's different
                    param.requires_grad = True
                    unfrozen_params_count += 1
                logger.debug(f"Parameter '{name}' set to requires_grad=True.")
            else:
                if param.requires_grad: # Only change if it's different
                    param.requires_grad = False
                    frozen_params_count += 1
                logger.debug(f"Parameter '{name}' set to requires_grad=False.")

        logger.info(f"Model parameters configured: {unfrozen_params_count} unfrozen, {frozen_params_count} frozen (out of {total_params} total).")


    def step(self):
        """
        Performs a single optimization step (parameter update).
        Delegates to the base optimizer's step method.
        """
        self.optimizer.step()

    def zero_grad(self):
        """
        Clears the gradients of all optimized parameters.
        Delegates to the base optimizer's zero_grad method.
        """
        self.optimizer.zero_grad()

    def state_dict(self) -> Dict[str, Any]:
        """
        Returns a serializable dictionary containing the current state of the optimizer.
        Delegates to the base optimizer's state_dict method.
        """
        return self.optimizer.state_dict()

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """
        Loads the optimizer's state from a state_dict.
        Delegates to the base optimizer's load_state_dict method.

        Args:
            state_dict (Dict[str, Any]): A dictionary containing the optimizer's state.
        """
        self.optimizer.load_state_dict(state_dict)

