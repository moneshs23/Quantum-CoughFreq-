from __future__ import annotations

import math
from typing import Tuple

import numpy as np
import torch
from qiskit import QuantumCircuit
from qiskit.circuit.library import RealAmplitudes, ZZFeatureMap
from qiskit.quantum_info import SparsePauliOp, Statevector
from torch import nn


class QuantumTBEstimator:
    def __init__(self, num_qubits: int = 4, encoding: str = "zz") -> None:
        self.num_qubits = num_qubits
        self.encoding = encoding
        self.ansatz = RealAmplitudes(num_qubits=num_qubits, reps=2, entanglement="full")
        self.measurement = SparsePauliOp.from_list([("Z" + "I" * (num_qubits - 1), 1.0)])
        self.parameter_count = self.ansatz.num_parameters

    def _feature_circuit(self, features: np.ndarray) -> QuantumCircuit:
        features = np.asarray(features, dtype=np.float64)
        if features.shape[0] != self.num_qubits:
            raise ValueError(f"Expected {self.num_qubits} features, received {features.shape[0]}.")

        if self.encoding == "angle":
            circuit = QuantumCircuit(self.num_qubits)
            for index, angle in enumerate(features):
                circuit.ry(float(angle), index)
                circuit.rz(float(angle) / 2.0, index)
            return circuit

        feature_map = ZZFeatureMap(feature_dimension=self.num_qubits, reps=1)
        assigned_map = feature_map.assign_parameters(features.tolist())
        return assigned_map.decompose()

    def tb_probability(self, features: np.ndarray, weights: np.ndarray) -> float:
        circuit = QuantumCircuit(self.num_qubits)
        circuit.compose(self._feature_circuit(features), inplace=True)
        circuit.compose(self.ansatz.assign_parameters(np.asarray(weights, dtype=np.float64).tolist()), inplace=True)

        state = Statevector.from_instruction(circuit)
        expectation = float(np.real(state.expectation_value(self.measurement)))
        return float((1.0 - expectation) / 2.0)

    def batch_tb_probabilities(self, batch_features: np.ndarray, weights: np.ndarray) -> np.ndarray:
        probabilities = [self.tb_probability(sample, weights) for sample in batch_features]
        return np.asarray(probabilities, dtype=np.float32)


class QuantumExpectationFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, weights: torch.Tensor, estimator: QuantumTBEstimator) -> torch.Tensor:
        probabilities = estimator.batch_tb_probabilities(
            batch_features=inputs.detach().cpu().numpy(),
            weights=weights.detach().cpu().numpy(),
        )
        ctx.estimator = estimator
        ctx.save_for_backward(inputs.detach(), weights.detach())
        return torch.from_numpy(probabilities).to(device=inputs.device, dtype=inputs.dtype)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, None]:
        inputs, weights = ctx.saved_tensors
        estimator = ctx.estimator
        epsilon = 1e-3

        base_inputs = inputs.cpu().numpy()
        base_weights = weights.cpu().numpy()
        grad_output_np = grad_output.detach().cpu().numpy().reshape(-1)

        grad_inputs = np.zeros_like(base_inputs, dtype=np.float32)
        for batch_index in range(base_inputs.shape[0]):
            for feature_index in range(base_inputs.shape[1]):
                plus = base_inputs.copy()
                minus = base_inputs.copy()
                plus[batch_index, feature_index] += epsilon
                minus[batch_index, feature_index] -= epsilon

                forward_plus = estimator.batch_tb_probabilities(plus, base_weights)[batch_index]
                forward_minus = estimator.batch_tb_probabilities(minus, base_weights)[batch_index]
                grad_inputs[batch_index, feature_index] = (
                    grad_output_np[batch_index] * (forward_plus - forward_minus) / (2.0 * epsilon)
                )

        grad_weights = np.zeros_like(base_weights, dtype=np.float32)
        for weight_index in range(base_weights.shape[0]):
            plus_weights = base_weights.copy()
            minus_weights = base_weights.copy()
            plus_weights[weight_index] += epsilon
            minus_weights[weight_index] -= epsilon

            forward_plus = estimator.batch_tb_probabilities(base_inputs, plus_weights)
            forward_minus = estimator.batch_tb_probabilities(base_inputs, minus_weights)
            parameter_gradient = (forward_plus - forward_minus) / (2.0 * epsilon)
            grad_weights[weight_index] = float(np.sum(grad_output_np * parameter_gradient))

        grad_input_tensor = torch.from_numpy(grad_inputs).to(device=inputs.device, dtype=inputs.dtype)
        grad_weight_tensor = torch.from_numpy(grad_weights).to(device=weights.device, dtype=weights.dtype)
        return grad_input_tensor, grad_weight_tensor, None


class TorchQuantumLayer(nn.Module):
    def __init__(self, num_qubits: int = 4, encoding: str = "zz") -> None:
        super().__init__()
        self.estimator = QuantumTBEstimator(num_qubits=num_qubits, encoding=encoding)
        initial_weights = torch.zeros(self.estimator.parameter_count, dtype=torch.float32)
        self.quantum_weights = nn.Parameter(initial_weights)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        bounded_inputs = torch.tanh(inputs / math.pi) * math.pi
        probabilities = QuantumExpectationFunction.apply(bounded_inputs, self.quantum_weights, self.estimator)
        return probabilities.unsqueeze(-1)
