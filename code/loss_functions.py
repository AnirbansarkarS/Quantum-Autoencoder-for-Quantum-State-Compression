from qiskit.quantum_info import state_fidelity, partial_trace, DensityMatrix

def fidelity_loss(original_statevector, circuit_with_params):
    """
    Computes loss based on 1 - fidelity(original, reconstructed).
    
    Args:
        original_statevector (Statevector): The input state.
        circuit_with_params (QuantumCircuit): The circuit with bound parameters.
        
    Returns:
        float: 1 - Fidelity
    """
    # This assumes we are evaluating the full reconstruction
    # In a QAE training, we often minimize the trash population.
    pass

def trash_population_loss(statevector, num_trash):
    """
    Computes the population of the trash qubits in the excited state.
    Goal is to minimize this to 'push' information into latent qubits.
    
    Args:
        statevector (Statevector): State after applying the encoder U(θ).
        num_trash (int): Number of trash qubits (usually the last qubits).
        
    Returns:
        float: Population of |1> in trash qubits.
    """
    # Total qubits
    num_qubits = statevector.num_qubits
    latent_qubits = num_qubits - num_trash
    
    # Trace out the latent qubits to get the reduced density matrix of the trash qubits
    # Qiskit uses little-endian qubit ordering (q0 is index 0)
    # If latent qubits are 0...k-1 and trash are k...n-1:
    trash_indices = list(range(latent_qubits, num_qubits))
    
    # To get density matrix of trash, we trace out latent qubits (0 to latent_qubits-1)
    latent_indices = list(range(0, latent_qubits))
    rho_trash = partial_trace(statevector, latent_indices)
    
    # We want trash state to be |0...0>
    # The probability of being in |0...0> is <0...0|rho_trash|0...0>
    # So loss = 1 - P(trash == |0...0>)
    
    # Density matrix of the zero state for num_trash qubits
    zero_state = DensityMatrix.from_label('0' * num_trash)
    fid = state_fidelity(rho_trash, zero_state)
    
    return 1 - fid
