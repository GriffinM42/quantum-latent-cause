import numpy as np
import numpy.linalg as lin
import math

# Matrix Functions
def sqrt(A: np.NDArray[any]):
    """Takes the square root of a matrix

    Parameters
    ----------
    A : matrix

    Returns
    -------
    matrix
        a matrix B such that B*B = A
    """
    eigvals, eigvecs = lin.eig(A)
    eigvals = np.sqrt(eigvals)
    return np.matmul(np.matmul(eigvecs, np.diag(eigvals)), lin.pinv(eigvecs))

def log(A: np.NDArray[any]):
    """Takes the natural logarithm of a matrix

    Parameters
    ----------
    A : matrix

    Returns
    -------
    matrix
        ln(A)
    """
    eigvals, eigvecs = lin.eig(A)
    eigvals = np.log(eigvals)
    return np.matmul(np.matmul(eigvecs, np.diag(eigvals)), lin.pinv(eigvecs))

def exp(A: np.NDArray[any]):
    """Takes the exponential of a matrix

    Parameters
    ----------
    A : matrix

    Returns
    -------
    matrix
        e^A
    """
    eigvals, eigvecs = lin.eig(A)
    eigvals = np.exp(eigvals)
    return np.matmul(np.matmul(eigvecs, np.diag(eigvals)), lin.pinv(eigvecs))

def trace_dist(A: np.NDArray[any], B: np.NDArray[any]):
    """Calculates the trace distance of two matrices

    Parameters
    ----------
    A : matrix
    B: matrix

    Returns
    -------
    float
        trace distance D(A, B)
    """
    mat = A - B
    mat_dag = np.conjugate(np.transpose(mat))
    return 0.5 * np.trace(sqrt(np.matmul(mat_dag, mat)))
    

# Related Objects
class QProblem:
    def __init__(self, esti_state: np.NDArray[any], dx: int, dy: int, dz: int):
        self.esti_state = esti_state
        self.dx = dx
        self.dy = dy
        self.dz = dz

class QGraphResult:
    def __init__(self, result_message: str, candidate_entropies: list[QWitness],
                 witnesses: list[QWitness]):
        self.result_message = result_message
        self.candidate_entropies = candidate_entropies
        self.witnesses = witnesses

        if self.candidate_entropies is not None:
            self.candidate_entropies.sort(key=lambda x:x.entrop_z)
        if self.witnesses is not None:
            self.witnesses.sort(key=lambda x:x.cmi)
    
    def get_optimal_witness(self):
        if self.candidate_entropies is not None and len(self.candidate_entropies) > 0:
            return min(self.candidate_entropies, key=lambda x:x.entrop_z)
        else:
            return None
    
    def get_min_cmi(self):
        if self.witnesses is not None:
            return min(self.witnesses, key=lambda x:x.cmi).cmi
        else:
            return None

    def get_max_cmi(self):
        if self.witnesses is not None:
            return max(self.witnesses, key=lambda x:x.cmi).cmi
        else:
            return None
    
    def get_median_cmi(self):
        if self.witnesses is not None:
            self.witnesses.sort(key=lambda x:x.cmi)
            return self.witnesses[math.floor(len(self.witnesses)/2)].cmi
        else:
            return None
    
    def get_min_entrop_z(self):
        if self.witnesses is not None:
            return min(self.witnesses, key=lambda x:x.entrop_z).entrop_z
        else:
            return None

    def get_max_entrop_z(self):
        if self.witnesses is not None:
            return max(self.witnesses, key=lambda x:x.entrop_z).entrop_z
        else:
            return None
    
    def get_median_entrop_z(self):
        if self.witnesses is not None:
            self.witnesses.sort(key=lambda x:x.entrop_z)
            med = self.witnesses[math.floor(len(self.witnesses)/2)].entrop_z
            self.witnesses.sort(key=lambda x:x.cmi)
            return med
        else:
            return None

    def get_percent_markov(self):
        if self.witnesses is not None:  
            return len(self.candidate_entropies)/len(self.witnesses)
        else:
            return 0.0

class QWitness:
    def __init__(self, penalty, state, cmi, entrop_z):
        self.penalty = penalty
        self.state = state
        self.cmi = cmi
        self.entrop_z = entrop_z

# Trace Operations
def tr_x(p_xyz: np.NDArray[any], dx: int, dy: int, dz: int):
    """Traces system x out of p_xyz via partial trace

    Parameters
    ----------
    p_xyz : matrix
    dx: int
        The dimension of system x
    dy: int
        The dimension of system y
    dz: int
        The dimension of system z

    Returns
    -------
    matrix
        p_yz = tr_x(p_xyz)
    """
    return np.reshape(np.trace(p_xyz.reshape(dx, dy, dz, dx, dy, dz), axis1=0, axis2=3), (dy*dz, dy*dz))

def tr_y(p_xyz: np.NDArray[any], dx: int, dy: int, dz: int):
    """Traces system y out of p_xyz via partial trace

    Parameters
    ----------
    p_xyz : matrix
    dx: int
        The dimension of system x
    dy: int
        The dimension of system y
    dz: int
        The dimension of system z

    Returns
    -------
    matrix
        p_xz = tr_y(p_xyz)
    """
    return np.reshape(np.trace(p_xyz.reshape(dx, dy, dz, dx, dy, dz), axis1=1, axis2=4), (dx*dz, dx*dz))

def tr_z(p_xyz: np.NDArray[any], dx: int, dy: int, dz: int):
    """Traces system z out of p_xyz via partial trace

    Parameters
    ----------
    p_xyz : matrix
    dx: int
        The dimension of system x
    dy: int
        The dimension of system y
    dz: int
        The dimension of system z

    Returns
    -------
    matrix
        p_xy = tr_z(p_xyz)
    """
    return np.reshape(np.trace(p_xyz.reshape(dx, dy, dz, dx, dy, dz), axis1=2, axis2=5), (dx*dy, dx*dy))

def tr_yz(p_xyz: np.NDArray[any], dx: int, dy: int, dz: int):
    """Traces system y and z out of p_xyz via partial trace

    Parameters
    ----------
    p_xyz : matrix
    dx: int
        The dimension of system x
    dy: int
        The dimension of system y
    dz: int
        The dimension of system z

    Returns
    -------
    matrix
        p_x = tr_yz(p_xyz)
    """
    p_xy = tr_z(p_xyz, dx, dy, dz)
    return np.reshape(np.trace(p_xy.reshape(dx, dy, dx, dy), axis1=1, axis2=3), (dx, dx))

def tr_xz(p_xyz: np.NDArray[any], dx: int, dy: int, dz: int):
    """Traces system x and z out of p_xyz via partial trace

    Parameters
    ----------
    p_xyz : matrix
    dx: int
        The dimension of system x
    dy: int
        The dimension of system y
    dz: int
        The dimension of system z

    Returns
    -------
    matrix
        p_y = tr_xz(p_xyz)
    """
    p_xy = tr_z(p_xyz, dx, dy, dz)
    return np.reshape(np.trace(p_xy.reshape(dx, dy, dx, dy), axis1=0, axis2=2), (dy, dy))

def tr_xy(p_xyz: np.NDArray[any], dx: int, dy: int, dz: int):
    """Traces system x and y out of p_xyz via partial trace

    Parameters
    ----------
    p_xyz : matrix
    dx: int
        The dimension of system x
    dy: int
        The dimension of system y
    dz: int
        The dimension of system z

    Returns
    -------
    matrix
        p_z = tr_xy(p_xyz)
    """
    p_xz = tr_y(p_xyz, dx, dy, dz)
    return np.reshape(np.trace(p_xz.reshape(dx, dz, dx, dz), axis1=0, axis2 = 2), (dz, dz))

# Conditional State Calculations
def get_zlx(p_xyz: np.NDArray[any], dx: int, dy: int, dz: int):
    """Calculates the conditional state of z given x (p_zlx)

    Parameters
    ----------
    p_xyz : matrix
    dx: int
        The dimension of system x
    dy: int
        The dimension of system y
    dz: int
        The dimension of system z

    Returns
    -------
    matrix
        p_zlx
    """
    p_xz = tr_y(p_xyz, dx, dy, dz)
    proj = np.kron(lin.pinv(sqrt(tr_yz(p_xyz, dx, dy, dz))), np.eye(dz))
    return np.matmul(np.matmul(proj, p_xz), proj)

def get_zly(p_xyz: np.NDArray[any], dx: int, dy: int, dz: int):
    """Calculates the conditional state of z given y (p_zly)

    Parameters
    ----------
    p_xyz : matrix
    dx: int
        The dimension of system x
    dy: int
        The dimension of system y
    dz: int
        The dimension of system z

    Returns
    -------
    matrix
        p_zly
    """
    p_yz = tr_x(p_xyz, dx, dy, dz)
    proj = np.kron(lin.pinv(sqrt(tr_xz(p_xyz, dx, dy, dz))), np.eye(dz))
    return np.matmul(np.matmul(proj, p_yz), proj)
    
# Regularized Marginal Calculations
def R_c(p_cz: np.NDArray[any], dc: int, dz: int, log_reg: float):
    """Mixes p_cz with a uniform distribution on z by a factor of the log regularization parameter

    Parameters
    ----------
    p_cz : matrix
        The joint density matrix of systems c and z (c being either x or y)
    dc: int
        The dimension of system c (c being either x or y)
    dz: int
        The dimension of system z
    log_reg: float
        The log regularization parameter

    Returns
    -------
    matrix
        The resultant density operator
    """
    return ((1-log_reg) * p_cz) + (log_reg * np.kron(np.eye(dc), np.eye(dz)/dz))

def R_z(p_z: np.NDArray[any], dz: int, log_reg: float):
    """Mixes p_z with a uniform distribution on z by a factor of the log regularization parameter

    Parameters
    ----------
    p_z : matrix
        The density matrix of system z
    dz: int
        The dimension of system z
    log_reg: float
        The log regularization parameter

    Returns
    -------
    matrix
        The resultant density operator
    """
    return ((1-log_reg) * p_z) + (log_reg/dz * np.eye(dz))

# Lifted Operators
def L_x(A: np.NDArray[any], dx: int, dy: int, dz: int):
    """Lifts a state z|x to xyz by tensoring with the identity on y

    Parameters
    ----------
    A : matrix
        The state z|x
    dx: int
        The dimension of system x
    dy: int
        The dimension of system y
    dz: int
        The dimension of system z

    Returns
    -------
    matrix
        The original state A tensored with the identity on the y system
    """
    A = np.kron(np.eye(dy), A)

    # Swap yxz to xyz
    A_1 = np.zeros(np.shape(A)).astype(np.complex128)
    for x in range(dx):
        for y in range(dy):
            for z in range(dz):
                for i in range(dx*dy*dz):
                    A_1[x*(dy*dz) + y*dz + z][i] = A[x*dz + y*(dx*dz) + z][i]
    
    A_2 = np.zeros(np.shape(A)).astype(np.complex128)
    for x in range(dx):
        for y in range(dy):
            for z in range(dz):
                for i in range(dx*dy*dz):
                    A_2[i][x*(dy*dz) + y*dz + z] = A_1[i][x*dz + y*(dx*dz) + z]

    return A_2

def L_y(A: np.NDArray[any], dx: int):
    """Lifts a state z|y to xyz by tensoring with the identity on x

    Parameters
    ----------
    A : matrix
        The state z|y
    dx: int
        The dimension of system x

    Returns
    -------
    matrix
        The original state A tensored with the identity on the x system
    """
    return np.kron(np.eye(dx), A)

def L_z(A: np.NDArray[any], dx: int, dy: int):
    """Lifts a state z to xyz by tensoring with the identity on x and y

    Parameters
    ----------
    A : matrix
        The state z
    dx: int
        The dimension of system x
    dy: int
        The dimension of system y

    Returns
    -------
    matrix
        The original state A tensored with the identity on the x and y systems
    """
    return np.kron(np.eye(dx*dy), A)

# Conditional Normalization
def condit_norm(R: np.NDArray[any], dx: int, dy: int, dz: int):
    """Calculates the conditional state of system z given systems x and y

    Parameters
    ----------
    R : matrix
        The joint density matrix of systems x, y, and z
    dx: int
        The dimension of system x
    dy: int
        The dimension of system y
    dz: int
        The dimension of system z

    Returns
    -------
    matrix
        The resultant conditional density operator z|xy
    """
    M_r = tr_z(R, dx, dy, dz)
    proj = np.kron(lin.pinv(sqrt(M_r)), np.eye(dz))
    return np.matmul(np.matmul(proj, R), proj)

# Entropy and Mutual Information Calculations
def vn_entropy(p_c: np.NDArray[any]):
    """Calculates von Neumann entropy of system c

    Parameters
    ----------
    p_c : matrix
        The density matrix of system c

    Returns
    -------
    float
        The von Neumann entropy of system c
    """
    eigvals = lin.eig(p_c)[0]
    vn = 0
    # Should we take the norm or just use the real component?
    for val in eigvals:
        if val != 0:
            vn += -1 * lin.norm(val) * math.log(lin.norm(val), 2)
    
    return vn

def mi_xy(p_xy: np.NDArray[any], dx: int, dy: int):
    """Calculates the mutual information of systems x and y

    Parameters
    ----------
    p_xy : matrix
        The joint density matrix of systems x and y
    dx: int
        The dimension of system x
    dy: int
        The dimension of system y

    Returns
    -------
    float
        The mutual information of x and y
    """
    p_x = np.trace(p_xy.reshape(dx, dy, dx, dy), axis1=1, axis2=3)
    p_y = np.trace(p_xy.reshape(dx, dy, dx, dy), axis1=0, axis2=2)

    return vn_entropy(p_x) + vn_entropy(p_y) - vn_entropy(p_xy)

def mi_xylz(p_xyz: np.NDArray[any], dx: int, dy: int, dz: int):
    """Calculates the conditional mutual information of systems x and y given system z

    Parameters
    ----------
    p_xyz : matrix
        The joint density matrix of systems x, y, and z
    dx: int
        The dimension of system x
    dy: int
        The dimension of system y
    dz: int
        The dimension of system z

    Returns
    -------
    float
        The mutual information of x and y given z
    """
    vn_xz = vn_entropy(tr_y(p_xyz, dx, dy, dz))
    vn_yz = vn_entropy(tr_x(p_xyz, dx, dy, dz))
    vn_z = vn_entropy(tr_xy(p_xyz, dx, dy, dz))
    vn_xyz = vn_entropy(p_xyz)

    return vn_xz + vn_yz - vn_z - vn_xyz

# Random Conditional Operator Initialization
def initC(dx: int, dy: int, dz: int):
    """Generates a pseudo random positive operator C such that C is positive semidefinite and
    tracing out system z results in the identity over systems x and y

    Parameters
    ----------
    dx: int
        The dimension of system x
    dy: int
        The dimension of system y
    dz: int
        The dimension of system z

    Returns
    -------
    matrix
        A pseudo random conditional density matrix z|xy
    """
    # Generate random matrix
    A = np.random.rand(dx*dy*dz, dx*dy*dz)
    # Make A a positive matrix
    A = np.matmul(np.transpose(np.conjugate(A)), A)
    # Give A trace 1
    tr_A = np.trace(A)
    A = A/tr_A

    #Condition on X and Y
    p_xy = tr_z(A, dx, dy, dz)
    proj = np.kron(lin.pinv(sqrt(p_xy)), np.eye(dz))

    return np.matmul(np.matmul(proj, A), proj)

# Latent Search
def QLatentSearch(problem: QProblem, smoothing: float, damping: float, log_reg: float, penalty: float, n: int):
    """Heuristically searches for a stable point of the trade-off between the von Neumann
    entropy of the hidden common cause and mutual information between the two observed systems

    Parameters
    ----------
    problem: QProblem
        The joint density matrix for the estimated state of the observed systems, the dimensions of systems x and y, and the
        maximum dimension for system z (hidden common cause)
    smoothing: float
        The factor to which the estimated state should be smoothed in order to stabilize
        the density matrix
    damping: float
        The factor to which the newly calculated iteration of C_z|xy contributes to the updated
        version of C_z|xy
    log_reg: float
        The log regularization parameter
    penalty: float
        The factor on the von Neumann entropy of the hidden common cause in the objective
        function
    n: int
        The number of iterations that the algorithm will perform to find a stable point

    Returns
    -------
    (matrix, float, float)
        - A proposed joint density matrix for systems x, y, and z denoting an estimated stable
        point on the objective function
        - The conditional mutual information of systems x and y given system z for the proposed
        joint state
        - The von Neumann entropy of the proposed system z
    """
    esti_state = problem.esti_state
    dx = problem.dx
    dy = problem.dy
    dz = problem.dz

    # Smooth the estimated state
    smooth_esti = ((1-smoothing) * esti_state) + ((smoothing/(dx*dy)) * np.eye(dx*dy))

    # Initialize random state z|xy
    C_zlxy = initC(dx, dy, dz)
    
    for i in range(n):
        #Form p_xyz
        condit_projector = np.kron(sqrt(smooth_esti), np.eye(dz))
        p_xyz = np.matmul(np.matmul(condit_projector, C_zlxy), condit_projector)

        #Compute C_zlx, C_zly, & p_z
        C_zlx = get_zlx(p_xyz, dx, dy, dz)
        C_zly = get_zly(p_xyz, dx, dy, dz)
        p_z = tr_xy(p_xyz, dx, dy, dz)

        #Compute R
        Lx = L_x(log(R_c(C_zlx, dx, dz, log_reg)), dx, dy, dz)
        Ly = L_y(log(R_c(C_zly, dy, dz, log_reg)), dx)
        Lz = L_z(log(R_z(p_z, dz, log_reg)), dx, dy)
        R_xyz = exp(Lx + Ly + (penalty-1)*Lz)

        #Compute C~ and C
        C_norm = condit_norm(R_xyz, dx, dy, dz)
        C_zlxy = ((1-damping)*C_zlxy) + (damping * C_norm)

    #Form p_xyz
    condit_projector = np.kron(sqrt(smooth_esti), np.eye(dz))
    p_xyz = np.matmul(np.matmul(condit_projector, C_zlxy), condit_projector)
    
    return QWitness(penalty, p_xyz, mi_xylz(p_xyz, dx, dy, dz), vn_entropy(tr_xy(p_xyz, dx, dy, dz)))

# Common Entropy
def QCommonEntropy(problem: QProblem, penalties: list[float], tolerance: float, smoothing: float, damping: float,
                    log_reg: float, n: int, print_iter = False):
    """Heuristically calculates the common entropy for the joint state of systems x and y

    Parameters
    ----------
    problem: QProblem
        The joint density matrix for the estimated state of the observed systems, the dimensions of systems x and y, and the
        maximum dimension for system z (hidden common cause)
    penalties: float array
        An array of penalty values to be used for QLatentSearch trials
    tolerance: float
        The threshold of conditional mutual information for a witness to be considered Markovizing
    smoothing: float
        The factor to which the estimated state should be smoothed in order to stabilize
        the density matrix
    damping: float
        The damping factor to be used in QLatentSearch
    log_reg: float
        The log regularization parameter to be used in QLatentSearch
    n: int
        The number of iterations that the QLatentSearch will perform to find a candidate witness
    print_iter: boolean
        Print each penalty value as its corresponding witness is calculated

    Returns
    -------
    tuple[float, matrix, float, float]
        - The penalty used in the best iteration of QLatentSearch
        - A proposed joint density matrix for systems x, y, and z denoting an estimated stable
        point on the objective function
        - The conditional mutual information of systems x and y given system z for the proposed
        joint state
        - The von Neumann entropy of the proposed system z
    list[tuple[float, matrix, float, float]]
        - A list of tuples representing all possible markovizing candidates, including the common
        entropy
    """

    # Iterates through penalty values to generate possible witnesses
    # Note: Assumes penalty values can be repeated, to try multiple randomizations,
    # and uses a seperate index
    witnesses = []
    for penalty in penalties:
        if print_iter:
            print(f'\r\33[KCurrent penalty: {penalty}', end='')

        i = 0
        # Allow multiple attempts for SVD to converge
        while True:
            try:
                witness = QLatentSearch(problem, smoothing, damping, log_reg, penalty, n)
                break
            except lin.LinAlgError:
                i += 1
                if i < 10:
                    print("Error converging, resetting problem")
                else:
                    print("Repeated error converging. Quitting")
                    quit()
                
        witnesses.append(witness)
    if print_iter:
        print("\n")
    # Finds minimum markovizing entropy value
    common_entropy = None
    candidates = []
    for witness_iter in witnesses:
        if (witness_iter.cmi <= tolerance):
            candidates.append(witness_iter)
            if((common_entropy is None) or (witness_iter.entrop_z < common_entropy.entrop_z)):
                common_entropy = witness_iter
    candidates.sort(key=lambda x:x.entrop_z)
    witnesses.sort(key=lambda x:x.cmi)
    
    return common_entropy, candidates, witnesses

# Infer Graph
def QInferGraph(problem: QProblem, penalties: list[float], tolerance: float, entrop_thresh: float, extern_thresh: float, dep_gate: float,
                 smoothing: float, damping: float, log_reg: float, n: int, null_fam: list[QProblem], sig_lvl: float, print_iter = False):
    """Heuristically infers the causal structure of two observed quantum systems base on common 
    entropy (the minimum entropy of a Markovizing hidden common cause)

    Parameters
    ----------
    problem: QProblem
        The joint density matrix for the estimated state of the observed systems, the dimensions of systems x and y, and the
        maximum dimension for system z (hidden common cause)
    penalties: float array
        An array of penalty values to be used for QLatentSearch trials
    tolerance: float
        The threshold of conditional mutual information for a witness to be considered Markovizing
    entrop_thresh: float
        The factor of the minimum entropy of the observed systems to use as a decision threshold if
        no external threshold is given
    extern_thresh: float
        The decision threshold for if the common entropy indicates the latent graph. Set to None if
        using entrop_thresh
    dep_gate: float
        Threshold for the mutual information of the observed systems to determine if there is enough
        correlation to explain
    smoothing: float
        The factor to which the estimated state should be smoothed in order to stabilize
        the density matrix
    damping: float
        The damping factor to be used in QLatentSearch
    log_reg: float
        The log regularization parameter to be used in QLatentSearch
    n: int
        The number of iterations that the QLatentSearch will perform to find a candidate witness
    null_fam: list[QProblem]
        Set of null calibration problems
    sig_lvl: float
        Lower tail significance level for null calibration
        print_iter: boolean
        Print each penalty value as its corresponding witness is calculated

    Returns
    -------
    QGraphResult
        The decision indicating if a possible Markovizing latent variable was found, as well as
        other candidate Markovizing witnesses
    """
    esti_state = problem.esti_state
    dx = problem.dx
    dy = problem.dy
    
    # Smooth the estimated state
    smooth_esti = ((1-smoothing) * esti_state) + ((smoothing/(dx*dy)) * np.eye(dx*dy))

    # Check if x and y have enough correlation to need explanation
    if mi_xy(smooth_esti, dx, dy) <= dep_gate:
        result_message = "not latent (too little dependence)"
        return QGraphResult(result_message, None, None)
    
    # Calculate quantum common entropy
    common_entropy, candidates, witnesses = QCommonEntropy(problem, penalties, tolerance, smoothing, damping, log_reg, n, print_iter)

    # Calculate null family
    null_stat = []
    for prob in null_fam:
        c_entrop = QCommonEntropy(prob, penalties, tolerance, smoothing, damping, log_reg, n)[0]
        if c_entrop is not None:
            null_stat.append(c_entrop.entrop_z)
    null_stat.sort()

    # Determin lower-tail threshold
    lt_index = math.floor(len(null_fam)*sig_lvl)
    quantile = null_stat[lt_index] if lt_index < len(null_stat) else None
    
    # Calculate entropy threshold
    if(extern_thresh is None):
        p_x = np.trace(esti_state.reshape(dx, dy, dx, dy), axis1=1, axis2=3)
        p_y = np.trace(esti_state.reshape(dx, dy, dx, dy), axis1=0, axis2=2)
        extern_thresh = entrop_thresh * min(vn_entropy(p_x), vn_entropy(p_y))
    
    result = QGraphResult("", candidates, witnesses)

    # Determine if threshold is held
    if (common_entropy is not None) and (common_entropy.entrop_z <= extern_thresh) and ((quantile is None) or (common_entropy.entrop_z < quantile)):
        result.result_message =  f"\nlatent Markovizing witness"
    else:
        result.result_message =  f"\nnot latent (common entropy above threshold)"
        
    result.result_message += f"\n\nOptimal Witness:\npenalty: {result.get_optimal_witness().penalty}\nmi_xy|z: {result.get_optimal_witness().cmi}\ns_z: {result.get_optimal_witness().entrop_z}" if common_entropy is not None else ""
    result.result_message += f"\n\n% Markovizing: {result.get_percent_markov()*100}\nmin mi_xy|z: {result.get_min_cmi()}\nmedian mi_xy|z: {result.get_median_cmi()}\nmax mi_xy|z: {result.get_max_cmi()}"
    result.result_message += f"\n\nmin s_z: {result.get_min_entrop_z()}\nmedian s_z: {result.get_median_entrop_z()}\nmax s_z: {result.get_max_entrop_z()}"
    
    return result
    
def getMinAlpha(problem: QProblem, penalties: list[float], tolerance: float, entrop_thresh: float, extern_thresh: float, dep_gate: float,
                 smoothing: float, damping: float, log_reg: float, n: int, null_fam: list[QProblem], sig_lvl: float):
    result = QInferGraph(problem, penalties, tolerance, entrop_thresh, extern_thresh, dep_gate, smoothing, damping, log_reg, n, null_fam, sig_lvl)
    
    esti_state = problem.esti_state
    dx = problem.dx
    dy = problem.dy

    p_x = np.trace(esti_state.reshape(dx, dy, dx, dy), axis1=1, axis2=3)
    p_y = np.trace(esti_state.reshape(dx, dy, dx, dy), axis1=0, axis2=2)
    entrop = min(vn_entropy(p_x), vn_entropy(p_y))

    if result.get_optimal_witness() is None:
        return None
    else:
        return result.get_optimal_witness().entrop_z / entrop
    
    