from qiskit.quantum_info import state_fidelity, partial_trace, DensityMatrix, Statevector

def fidelity_loss(original_statevector, circuit_with_params):
    """
    Computes loss based on 1 - fidelity(original, reconstructed).
    """
    # This assumes we are evaluating the full reconstruction
    pass

def trash_population_loss(state_input, num_trash):
    """
    Computes the population of the trash qubits in the excited state.
    Goal is to minimize this to 'push' information into latent qubits.
    
    Args:
        state_input (Statevector or array): State after applying the encoder U(θ).
        num_trash (int): Number of trash qubits (usually the last qubits).
        
    Returns:
        float: Population of |1> in trash qubits.
    """
    # Ensure we have a Statevector object
    if not hasattr(state_input, 'num_qubits'):
        state_input = Statevector(state_input)
        
    num_qubits = state_input.num_qubits
    latent_qubits = num_qubits - num_trash
    
    # Trace out the latent qubits to get the reduced density matrix of the trash qubits
    latent_indices = list(range(0, latent_qubits))
    rho_trash = partial_trace(state_input, latent_indices)
    
    # We want trash state to be |0...0>
    # Density matrix of the zero state for num_trash qubits
    zero_state = DensityMatrix.from_label('0' * num_trash)
    fid = state_fidelity(rho_trash, zero_state)
    
    return 1 - fid
