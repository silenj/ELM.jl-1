module ELM

using DataFrames

using LinearAlgebra
import LinearAlgebra: pinv

export ExtremeLearningMachine
export fit!, predict

# Base code
include("base.jl")

end
