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

# Problem Object
class QProblem:
    def __init__(self, esti_state: np.NDArray[any], dx: int, dy: int, dz: int):
        self.esti_state = esti_state
        self.dx = dx
        self.dy = dy
        self.dz = dz

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
    
    return p_xyz, mi_xylz(p_xyz, dx, dy, dz), vn_entropy(tr_xy(p_xyz, dx, dy, dz))

# Common Entropy
def QCommonEntropy(problem: QProblem, penalties: list[float], tolerance: float, smoothing: float, damping: float,
                    log_reg: float, n: int):
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

    Returns
    -------
    (float, matrix, float, float)
        - The penalty used in the best iteration of QLatentSearch
        - A proposed joint density matrix for systems x, y, and z denoting an estimated stable
        point on the objective function
        - The conditional mutual information of systems x and y given system z for the proposed
        joint state
        - The von Neumann entropy of the proposed system z
    """

    # Iterates through penalty values to generate possible witnesses
    # Note: Assumes penalty values can be repeated, to try multiple randomizations,
    # and uses a seperate index
    witnesses = {}
    index = 0
    for penalty in penalties:
        p_xyz, mi, sz = QLatentSearch(problem, smoothing, damping, log_reg, penalty, n)
        witnesses[index] = (penalty, p_xyz, mi, sz)
        index += 1

    # Finds minimum markovizing entropy value
    common_entropy = None
    for witness in witnesses:
        if (witnesses[witness][2] <= tolerance) and ((common_entropy is None) or (witnesses[witness][3] < common_entropy[3])):
            common_entropy = witnesses[witness]
    
    return common_entropy

# Infer Graph
def QInferGraph(problem: QProblem, penalties: list[float], tolerance: float, entrop_thresh: float, extern_thresh: float, dep_gate: float,
                 smoothing: float, damping: float, log_reg: float, n: int, null_fam: list[QProblem], sig_lvl: float):
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

    Returns
    -------
    string
        The decision indicating if a possible Markovizing latent variable was found
    """
    esti_state = problem.esti_state
    dx = problem.dx
    dy = problem.dy
    
    # Smooth the estimated state
    smooth_esti = ((1-smoothing) * esti_state) + ((smoothing/(dx*dy)) * np.eye(dx*dy))

    # Check if x and y have enough correlation to need explanation
    if mi_xy(smooth_esti, dx, dy) <= dep_gate:
        return "not latent (too little dependence)"
    
    # Calculate quantum common entropy
    common_entropy = QCommonEntropy(problem, penalties, tolerance, smoothing, damping, log_reg, n)

    # Calculate null family
    null_stat = []
    for prob in null_fam:
        c_entrop = QCommonEntropy(prob, penalties, tolerance, smoothing, damping, log_reg, n)
        if c_entrop is not None:
            null_stat.append(c_entrop[3])
    null_stat.sort()

    # Determin lower-tail threshold
    lt_index = math.ceil(len(null_fam)*sig_lvl)-1
    quantile = null_stat[lt_index] if lt_index < len(null_stat) else None
    
    # Calculate entropy threshold
    if(extern_thresh is None):
        p_x = np.trace(esti_state.reshape(dx, dy, dx, dy), axis1=1, axis2=3)
        p_y = np.trace(esti_state.reshape(dx, dy, dx, dy), axis1=0, axis2=2)
        extern_thresh = entrop_thresh * min(vn_entropy(p_x), vn_entropy(p_y))
    
    # Determine if threshold is held
    if (common_entropy is not None) and (common_entropy[3] <= extern_thresh) and ((quantile is None) or (common_entropy[3] < quantile)):
        return f"latent Markovizing witness\n\npenalty: {common_entropy[0]}\nmi_xy|z: {common_entropy[2]}\ns_z: {common_entropy[3]}"
    else:
        return f"not latent (common entropy above threshold){f"\n\npenalty: {common_entropy[0]}\nmi_xy|z: {common_entropy[2]}\ns_z: {common_entropy[3]}" if common_entropy is not None else ""}"
    
    

    
    