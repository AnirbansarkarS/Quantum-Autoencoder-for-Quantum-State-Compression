# Phase 2 Theory Explainer: The Mechanics of Quantum Autoencoders

Phase 2 focuses on moving from theory to a working **Minimal Quantum Autoencoder (Core Build)**. Here, we implement the physical circuit, the quantum loss function, and the classical optimization loop.

---

## 1. The Architecture: Learning the Subspace

The core of a QAE is a **parameterized unitary transformation** $U(\theta)$. Unlike classical autoencoders that squeeze data through a bottleneck layer, the QAE *reshuffles* quantum information.

### Mathematical Goal
Map a state $|\psi\rangle$ on $n$ qubits into a state where $k$ qubits (latent) contain the information and $n-k$ qubits (trash) are in the $|0\rangle$ state.

$$ U(\theta)|\psi\rangle \approx |\text{compressed}\rangle_k \otimes |0\rangle_{n-k} $$

---

## 2. The Parameterized Unitary (Ansatz)

In this implementation, we use a **RealAmplitudes** ansatz. 
- **Layers:** Consists of alternating layers of single-qubit rotations (Ry) and entangling gates (CNOT).
- **Entanglement:** Full entanglement ensures that any correlations in the input state can be processed.
- **Differentiability:** While we use classical optimizers, this circuit defines a smooth landscape for optimization.

---

## 3. The Quantum-Specific Loss Function

We use the **Trash Qubit Population** as our training objective.

### Why not Fidelity here?
In real quantum hardware, measuring fidelity between an unknown input and an output requires expensive SWAP tests. However, measuring if a qubit is in the $|0\rangle$ or $|1\rangle$ state is a **standard projective measurement**.

### The Equation
$$ L(\theta) = 1 - P(\text{trash} = |0\rangle) = \langle 1 | \rho_{\text{trash}} | 1 \rangle $$

If the trash qubits are successfully "emptied" of information (i.e., they are in $|0\rangle$), then by the principle of information conservation in unitary evolution, all information must reside in the latent qubits.

---

## 4. The Hybrid Optimization Loop

The training is a **Variational Quantum Algorithm (VQA)**:

1. **Quantum Processor:** Prep input $|\psi\rangle$, apply $U(\theta)$, measure trash.
2. **Classical Processor:** Calculate loss, update $\theta$ using **COBYLA** (Constrained Optimization BY Linear Approximation).
3. **Repeat:** Until the trash population is minimized.

---

## 5. Verification: Bell State Compression

A key verification is compressing a **Bell State** (2 qubits) into 1 latent qubit.
- A Bell state is maximally entangled.
- Classical compression would fail to capture the coherence.
- The QAE learns a basis transformation that "disentangles" the Bell state, mapping it to a separable state where one qubit can be safely discarded.

---

**Next Step:** Run the training script to see the loss convergence in action.
