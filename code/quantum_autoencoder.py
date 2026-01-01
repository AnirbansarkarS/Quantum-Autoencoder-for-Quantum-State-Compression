import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import RealAmplitudes

class QuantumAutoencoder:
    """
    Implements a Quantum Autoencoder architecture.
    """
    
    def __init__(self, num_qubits, num_latent, reps=3, entanglement='full'):
        """
        Initialize the QAE.
        
        Args:
            num_qubits (int): Total number of qubits (input space).
            num_latent (int): Number of latent qubits (compressed space).
            reps (int): Number of repetitions for the variational ansatz.
            entanglement (str): Entanglement strategy.
        """
        self.num_qubits = num_qubits
        self.num_latent = num_latent
        self.num_trash = num_qubits - num_latent
        self.reps = reps
        self.entanglement = entanglement
        
        if num_latent >= num_qubits:
            raise ValueError("Number of latent qubits must be less than total qubits.")
            
        # Define the parameterized unitary (ansatz)
        self.ansatz = RealAmplitudes(
            num_qubits=self.num_qubits,
            reps=self.reps,
            entanglement=self.entanglement
        )
        
        self.num_parameters = self.ansatz.num_parameters
        
    def get_autoencoder_circuit(self, params=None):
        """
        Build the full autoencoder circuit.
        """
        qc = QuantumCircuit(self.num_qubits)
        bound_ansatz = self.ansatz.assign_parameters(params) if params is not None else self.ansatz
        qc.append(bound_ansatz, range(self.num_qubits))
        return qc

    def get_reconstruction_circuit(self, params):
        """
        Build the reconstruction circuit (U followed by U†).
        """
        qc = QuantumCircuit(self.num_qubits)
        ansatz_u = self.ansatz.assign_parameters(params)
        qc.append(ansatz_u, range(self.num_qubits))
        qc.append(ansatz_u.inverse(), range(self.num_qubits))
        return qc
