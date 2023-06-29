using Turing, DifferentialEquations, LinearAlgebra

@model function fitlv(data, prob)
    # Prior distributions.
    σ ~ InverseGamma(2, 3)
    α ~ truncated(Normal(1.5, 0.5); lower=0.5, upper=2.5)
    β ~ truncated(Normal(1.2, 0.5); lower=0, upper=2)
    γ ~ truncated(Normal(3.0, 0.5); lower=1, upper=4)
    δ ~ truncated(Normal(1.0, 0.5); lower=0, upper=2)

    # Simulate Lotka-Volterra model. 
    p = [α, β, γ, δ]
    predicted = solve(prob, Tsit5(); p=p, saveat=0.1)

    # Observations.
    for i in 1:length(predicted)
        data[:, i] ~ MvNormal(predicted[i], σ^2 * I)
    end

    return nothing
end

# Define Lotka-Volterra model.
function lotka_volterra(du, u, p, t)
    # Model parameters.
    α, β, γ, δ = p
    # Current state.
    x, y = u

    # Evaluate differential equations.
    du[1] = (α - β * y) * x # prey
    du[2] = (δ * x - γ) * y # predator

    return nothing
end

function make_lv_prob()
    # Define initial-value problem.
    u0 = [1.0, 1.0]
    p = [1.5, 1.0, 3.0, 1.0]
    tspan = (0.0, 10.0)
    prob = ODEProblem(lotka_volterra, u0, tspan, p)
    return prob, tspan
end

function make_lv_data()
    prob, tspan = make_lv_prob()
    sol = solve(prob, Tsit5(); saveat=0.1)
    odedata = Array(sol) + 0.8 * randn(size(Array(sol)))
    return odedata, prob
end

function debug_lv()
    odedata, prob = make_lv_data()
    model = fitlv(odedata, prob)

    mle_estimate = optimize(model, MLE())
    init_params = Iterators.repeated(mle_estimate.values.array)

    chain = Turing.sample(model, NUTS(), MCMCSerial(), 1000, 3;
                          init_params = init_params,
                          progress=false)
    
end

