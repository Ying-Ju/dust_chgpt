"""
    changepoint_mcmc(X_obs;
        N::Int=1000, 
        Tmax::Int,
        xi::Float64=log(2)/7,
        breaks::Int=2,
        n_iter::Int=5000,
        burn_in::Int=1000,
        thin::Int=1,
        prop_sd::NamedTuple=(theta1=0.05, theta2=0.05, tau=2.0)
    ) -> Matrix{Float64}

Perform Metropolis–Hastings to sample (θ₁, θ₂, τ), given a fixed observed
dust series X_obs over M = length(X_obs) intervals of length `breaks`.

Note: `simulate_lambda` (and its internal infection sim) is still used to
generate the proposal likelihood under each (θ₁, θ₂, τ).
"""
function changepoint_mcmc(
    X_obs::Vector{<:Integer};
    N::Int, 
    Tmax::Int,
    xi::Float64=log(2)/7,
    breaks::Int=2,
    n_iter::Int=5000,
    burn_in::Int=1000,
    thin::Int=1,
    prop_sd::NamedTuple=(theta1=0.05, theta2=0.05, tau=2.0)
)
    #M = length(X_obs)

    # allocate storage
    n_samps = (n_iter - burn_in) ÷ thin
    samples = Array{Float64}(undef, n_samps, 3)  # columns: θ₁, θ₂, τ
    logalphas = Array{Float64}(undef, n_samps)   # stores log acceptance ratios

    # initialize
    θ1_cur = 0.5                        # ∼Uniform(0,1)
    θ2_cur = 0.5 #rand(Uniform(θ1_cur, 1.0))    # ∼Uniform(θ1,1)
    τ_cur  = rand(1:(Tmax-1))             # integer jump‐point

    # initial log‐lik & log‐prior
    λ_cur        = simulate_lambda(
                        N      = N,        
                        Tmax   = Tmax,
                        theta1 = θ1_cur,
                        theta2 = θ2_cur,
                        tau    = τ_cur,
                        xi     = xi,
                        breaks = breaks
                   )

    loglik_cur   = loglik_binom_poisson_sum(X_obs, λ_cur, xi; δt=breaks)
    logprior_cur = log_prior(θ1_cur, θ2_cur, τ_cur; Tmax=Tmax)

    counter = 1
    accept_count = 0

    for iter in 1:n_iter
        
        # propose
        θ1_prop = rand(Normal(θ1_cur, prop_sd.theta1))
        θ2_prop = rand(Normal(θ2_cur, prop_sd.theta2))
        τ_prop  = τ_cur + rand(-prop_sd.tau:prop_sd.tau) #round(Int, rand(Normal(τ_cur, prop_sd.tau)))
        
        # quick rejects
        if θ1_prop ≤ 0 || θ1_prop ≥ 1 #|| θ2_prop ≥ 1 || θ2_prop ≤ 0
            continue
        elseif θ2_prop ≥ 1 # θ2_prop ≤ θ1_prop || 
            continue
        elseif τ_prop ≤ 0 || τ_prop ≥ Tmax 
            continue
        end

        # initial log‐lik & log‐prior
        λ_cur        = simulate_lambda(
                            N      = N,        
                            Tmax   = Tmax,
                            theta1 = θ1_cur,
                            theta2 = θ2_cur,
                            tau    = τ_cur,
                            xi     = xi,
                            breaks = breaks
                    )
        loglik_cur   = loglik_binom_poisson_sum(X_obs, λ_cur, xi; δt=breaks)
        logprior_cur = log_prior(θ1_cur, θ2_cur, τ_cur; Tmax=Tmax)

        # simulate λ under proposal
        λ_prop = simulate_lambda(
                      N      = N,
                      Tmax   = Tmax,
                      theta1 = θ1_prop,
                      theta2 = θ2_prop,
                      tau    = τ_prop,
                      xi     = xi,
                      breaks = breaks
                  )
        

        if any(!isfinite, λ_prop) || any(≤(0.0), λ_prop)
            continue
        end

        # compute loglik & logprior
        loglik_prop   = loglik_binom_poisson_sum(X_obs, λ_prop, xi; δt=breaks)
        logprior_prop = log_prior(θ1_prop, θ2_prop, τ_prop; Tmax=Tmax)


        # MH accept
        logα = (loglik_prop + logprior_prop) -
               (loglik_cur + logprior_cur)
        if log(rand()) < logα
            accept_count += 1
            θ1_cur, θ2_cur, τ_cur = θ1_prop, θ2_prop, τ_prop
            loglik_cur, logprior_cur = loglik_prop, logprior_prop
        end
        
        
        # store post‐burn & thinning
        if iter > burn_in && ((iter - burn_in) % thin == 0)
            samples[counter, :] = [θ1_cur, θ2_cur, τ_cur]
            logalphas[counter] = logα
            counter += 1
        end

        #if iter % 500 == 0
        #    @info "Iteration $iter/$n_iter"
        #end
    end
    
    accepted_rate = accept_count / n_iter
    return (samples = samples[1:(counter - 1), :],
        logalphas = logalphas[1:(counter - 1)],
        accepted_rate = accepted_rate)

end


"""
    shedding_growth(t, δ, ν, w, τ)

Growth‐phase shedding:
  λ(t) = exp( log(δ+1) * (t - w) / (ν - w) ) - 1
"""
function shedding_growth(t, δ, ν, w, τ)
    denom = ν - w
    if denom <= 1e-8
        # effectively no growth phase; treat as ~0 (or a tiny value)
        return 0.0
    end
    a = log(δ + 1) / denom
    val = exp(a * (t - w)) - 1
    return isfinite(val) && val ≥ 0 ? val : 0.0
end


"""
    shedding_decay(t, δ, ν, w, τ)

Decay‐phase shedding:
  λ(t) = exp( log(δ+1) * ((w + τ) - t) / ((w + τ) - ν) ) - 1
"""

function shedding_decay(t, δ, ν, w, τ)
    denom = (w + τ) - ν
    if denom <= 1e-8
        # effectively no decay phase
        return 0.0
    end
    a = log(δ + 1) / denom
    val = exp(a * ((w + τ) - t)) - 1
    return isfinite(val) && val ≥ 0 ? val : 0.0
end


"""
    compute_mu(a, b, δ, ν, w, τ)

Compute ∫ shedding(t) dt, where shedding is growth up to ν then decay,
and is zero outside [w, w+τ].
- δ = delta
- ν = nu
- w = onset
- τ = total_duration
"""
function compute_mu(a, b, δ, ν, w, τ)
    # outside support
    if b <= w || a >= w + τ
        #println("Issue 1")
        return 0.0
    end

    a2 = max(a, w)
    b2 = min(b, w + τ)
    if b2 <= a2
        #println("Issue 2")
        return 0.0
    end

    # clamp ν into [w, w+τ] to avoid impossible phase splits
    νc = clamp(ν, w, w + τ)

    integrand_growth = t -> shedding_growth(t, δ, νc, w, τ)
    integrand_decay  = t -> shedding_decay(t, δ, νc, w, τ)

    val = if b2 ≤ νc
        quadgk(integrand_growth, a2, b2; rtol=1e-6)[1]
    elseif a2 ≥ νc
        quadgk(integrand_decay, a2, b2; rtol=1e-6)[1]
    else
        g = quadgk(integrand_growth, a2, νc; rtol=1e-6)[1]
        d = quadgk(integrand_decay,  νc, b2; rtol=1e-6)[1]
        g + d
    end

    return (isfinite(val) && val ≥ 0) ? val : 0.0
end


"""
    compute_p(a, b, δ, ν, w, τ, ξ)

Compute 
p = ∫ e^{-ξ (b - s)} λ(s) ds} / ∫ λ(s) ds
where λ(s)= growth up to ν then decay, zero outside [w, w+τ].
- δ = delta
- ν = nu
- w = onset
- τ = total_duration
- ξ = xi
"""
function compute_p(a, b, δ, ν, w, τ, ξ)
    if b <= w || a >= w + τ
        return 0.0
    end

    a2 = max(a, w)
    b2 = min(b, w + τ)
    if b2 <= a2
        return 0.0
    end

    νc = clamp(ν, w, w + τ)

    # denominator: total shedding on [a2,b2]
    den = compute_mu(a2, b2, δ, νc, w, τ)
    if !(isfinite(den)) || den <= 1e-12
        return 0.0
    end

    # numerator
    integrand_growth = s -> exp(-ξ * (b2 - s)) * shedding_growth(s, δ, νc, w, τ)
    integrand_decay  = s -> exp(-ξ * (b2 - s)) * shedding_decay(s,  δ, νc, w, τ)

    num = if b2 ≤ νc
        quadgk(integrand_growth, a2, b2; rtol=1e-6)[1]
    elseif a2 ≥ νc
        quadgk(integrand_decay,  a2, b2; rtol=1e-6)[1]
    else
        g = quadgk(integrand_growth, a2, νc; rtol=1e-6)[1]
        d = quadgk(integrand_decay,  νc, b2; rtol=1e-6)[1]
        g + d
    end

    if !isfinite(num) || num < 0
        return 0.0
    end

    p = num / den

    # p should behave like a weight; keep it stable
    return clamp(p, 0.0, 1.0)
end


@inline logfactorial(n::Int) = loggamma(n + 1)

@inline function logpmf_poisson(k::Int, λ::Float64)
    if k < 0
        return -Inf
    end
    if λ == 0.0
        return (k == 0) ? 0.0 : -Inf
    elseif λ < 0.0 || !isfinite(λ)
        return -Inf
    end
    return -λ + k*log(λ) - logfactorial(k)
end


@inline function logpmf_binomial(k::Int, n::Int, p::Float64)
    if k < 0 || k > n || n < 0
        return -Inf
    end
    if !isfinite(p) || p < 0.0 || p > 1.0
        return -Inf
    end
    if p == 0.0
        return (k == 0) ? 0.0 : -Inf
    elseif p == 1.0
        return (k == n) ? 0.0 : -Inf
    end
    return (logfactorial(n) - logfactorial(k) - logfactorial(n-k)
            + k*log(p) + (n-k)*log1p(-p))
end

"""
    loglik_binom_poisson_sum(X_obs, λ_vec, ξ; δt=2)

Compute the log‐likelihood of the observed dust counts
    X_obs = [X_obs[1], X_obs[2], ..., X_obs[M]]
    λ_vec = [λ_vec[1], λ_vec[2], ..., λ_vec[M]]
    ξ = decay rate
    δt = time interval length (default=2)
"""

function loglik_binom_poisson_sum(
    X_obs::AbstractVector{<:Integer},
    λ_vec::AbstractVector{<:Real},
    ξ::Real; δt::Int = 2
)
    M = length(X_obs)
    @assert length(λ_vec) == M

    pdecay = exp(-ξ * δt)
    if !(isfinite(pdecay)) || pdecay < 0 || pdecay > 1
        return -Inf
    end

    loglik = 0.0
    X_prev = 0

    for l in 1:M
        x = Int(X_obs[l])
        if x < 0
            return -Inf
        end

        λ = float(λ_vec[l])
        if !(isfinite(λ)) || λ < 0
            return -Inf
        end

        kmin = max(0, x - X_prev)
        kmax = x

        m = -Inf
        s = 0.0

        @inbounds for k in kmin:kmax
            lp = logpmf_poisson(k, λ)
            lb = logpmf_binomial(x - k, X_prev, pdecay)
            v  = lp + lb
            
            # ignore zero-probability terms
            if v == -Inf
                continue
            end

            if v > m
                s = isfinite(m) ? (s * exp(m - v) + 1.0) : 1.0
                m = v
            else
                s += exp(v - m)
            end
        end
        
        # If all terms were -Inf, then the convolution sum is 0 => log is -Inf
        if m == -Inf
            return -Inf
        end

        loglik += m + log(s)
        X_prev = x
    end

    return loglik
end

"""
    log_prior(θ1, θ2, τ; Tmax=60)

Log‐prior that is 0 if
  0 < θ1 < 1,
  θ1 < θ2 < 2,
  0 < τ < Tmax
and –Inf otherwise.
"""

function log_prior(θ1::Real, θ2::Real, τ::Real; Tmax::Real=60)
    # Support checks
    #if !(0.0 < θ1 < 1.0) || !(θ1 < θ2 < 1.0) || !(1 <= τ <= Tmax - 1)
    if !(0.0 < θ1 < 1.0) || !(0.0 < θ2 < 1.0) || !(1 <= τ <= Tmax - 1)
        return -Inf
    end

    lp_θ1 = 0.0                      # Uniform(0,1):    log(1)
    lp_θ2 = 0.0 #-log(1.0 - θ1)           # Uniform(θ1,1):   log(1/(1−θ1))
    lp_τ  = -log(Float64(Tmax) - 2)  # Disc. Uniform:   log(1/(Tmax−2))

    return lp_θ1 + lp_θ2 + lp_τ
end


"""
    simulate_dust_observation(; N=1000, Tmax=100, theta1=0.5,
                              xi=log(2)/7, breaks=10,
                              theta2=nothing, t_star=nothing)

Simulate infection (via simulate_infection_process) and then
observe viral RNA in dust over intervals of length `breaks`.
Returns a NamedTuple with
  • observations::DataFrame  – columns: Interval, Start, End, Dust
  • infection_history        – same structure as simulate_infection_process
  • states                   – the 0/1 state matrix
"""
function simulate_dust_observation(; N::Int=1000,
                                   Tmax::Int=100,
                                   theta1::Float64=0.5,
                                   xi::Float64=log(2)/7,
                                   breaks::Int=10,
                                   theta2::Union{Nothing,Float64}=nothing,
                                   t_star::Union{Nothing,Int}=nothing)

    # 1) Set up observation times
    obs_times = collect(0:breaks:Tmax)
    M = length(obs_times) - 1
    X_Tt = zeros(Int, M)
    X_Tt_prev = 0

    # 2) Run the infection sim
    sim = simulate_infection_process(
             N = N, Tmax = Tmax,
             theta1 = theta1, theta2 = theta2,
             t_star = t_star
          )
    hist = sim.history

    # 3) Loop over intervals
    for ℓ in 1:M
        a, b = obs_times[ℓ], obs_times[ℓ+1]
        λsum = 0.0

        for i in 1:N
            for inf in hist[i]
                μ = compute_mu(a, b,
                               inf.delta_ik,
                               inf.v_ik,
                               inf.w_ik,
                               inf.tau_ikr)
                p = compute_p(a, b,
                              inf.delta_ik,
                              inf.v_ik,
                              inf.w_ik,
                              inf.tau_ikr,
                              xi)
                if isfinite(μ) && isfinite(p) && μ>0 && p>0
                    λsum += μ * p
                end
            end
        end

        # simulate counts
        Y1 = rand(Poisson(λsum))
        decay_prob = exp(-xi*(b - a))
        Y2 = rand(Binomial(X_Tt_prev, decay_prob))

        X_Tt[ℓ] = Y1 + Y2
        X_Tt_prev = X_Tt[ℓ]
    end

    # 4) Build a DataFrame of results
    df = DataFrame(
        Interval = [ "($(obs_times[j]),$(obs_times[j+1])]" for j in 1:M ],
        Start    = obs_times[1:(end-1)],
        End      = obs_times[2:end],
        Dust     = X_Tt,
    )

    return (
      observations     = df,
      infection_history = hist,
      states           = sim.states
    )
end


"""
    simulate_lambda(; N=1000,
                    Tmax=60,
                    theta1=0.2,
                    theta2=0.5,
                    tau=30,
                    xi=log(2)/7,
                    breaks=10)

For each interval of length `breaks` over `0:Tmax`, compute
lambda = sum_{i,k} mu_{ik}*p_{ik},
where
  mu_{ik} = ∫ shedding(t) dt, where shedding is growth up to ν then decay,
            and is zero outside [w, w+τ].
  p_{ik} = ∫ e^{-ξ (b - s)} λ(s) ds / ∫ λ(s) ds
          where λ(s)= growth up to ν then decay, zero outside [w, w+τ].
using the infection history from `simulate_infection_process`.
Returns a Vector{Float64} of length `M = floor(Tmax / breaks)`.
"""
function simulate_lambda(; N::Int=1000,
                          Tmax::Int=60,
                          theta1::Float64=0.2,
                          theta2::Float64=0.5,
                          tau::Int=30,
                          xi::Float64=log(2)/7,
                          breaks::Int=10)

    # 1) Observation grid
    obs_times = collect(0:breaks:Tmax)
    M = length(obs_times) - 1
    lambda_vec = zeros(Float64, M)

    # 2) Run infection sim (t_star = tau)
    sim = simulate_infection_process(
        N = N, Tmax = Tmax,
        theta1 = theta1,
        theta2 = theta2,
        t_star = tau
    )
    hist = sim.history

    # 3) Loop over intervals
    for ℓ in 1:M
        a, b = obs_times[ℓ], obs_times[ℓ+1]
        sumλ = 0.0

        for i in 1:N
            for inf in hist[i]
                μ = compute_mu(a, b,
                               inf.delta_ik,
                               inf.v_ik,
                               inf.w_ik,
                               inf.tau_ikr)
                p = compute_p(a, b,
                              inf.delta_ik,
                              inf.v_ik,
                              inf.w_ik,
                              inf.tau_ikr,
                              xi)
                if isfinite(μ) && isfinite(p) && μ ≥ 0 && p ≥ 0
                    sumλ += μ * p
                end
                #println("Interval $ℓ: i=$i, inf=$(inf), μ=$μ, p=$p, a = $a, b = $b")
            end
        end

        lambda_vec[ℓ] = sumλ
    end

    return lambda_vec
end


"""
    simulate_infection_process(
    ; N::Int=1000,
    Tmax::Int=60,
    theta1::Float64=0.5,
    theta2::Union{Nothing,Float64}=nothing,
    t_star::Union{Nothing,Int}=nothing
)

Simulate an infection process with N individuals over Tmax time steps.
"""
function simulate_infection_process(
    ; N::Int=1000,
    Tmax::Int=60,
    theta1::Float64=0.5,
    theta2::Union{Nothing,Float64}=nothing,
    t_star::Union{Nothing,Int}=nothing
)

    if theta2 !== nothing && t_star === nothing
        error("t_star must be provided if theta2 is used.")
    end

    states = zeros(Int, N, Tmax+1)
    current_state = fill(0, N)
    recover_time  = fill(typemax(Int), N)

    initial_infected = 5
    ids0 = randperm(N)[1:initial_infected]
    states[ids0, 1] .= 1
    current_state[ids0] .= 1

    history = [Vector{NamedTuple{(:w_ik,:tau_ikp,:tau_ikr,:delta_ik,:v_ik),
                                Tuple{Int,Float64,Float64,Float64,Float64}}}()
               for _ in 1:N]

    for i in ids0
        w_ik = 0
        tau_ikp = rand(truncated(Normal(4.2, 0.5), 1e-3, Inf))
        extra   = rand(truncated(Normal(7.3, 0.6), 1e-3, Inf))
        tau_ikr = tau_ikp + extra
        delta_ik = rand(Exponential(20.0))
        v_ik = w_ik + tau_ikp

        push!(history[i], (w_ik=w_ik, tau_ikp=tau_ikp, tau_ikr=tau_ikr,
                           delta_ik=delta_ik, v_ik=v_ik))

        recover_time[i] = w_ik + ceil(Int, tau_ikr)
    end

    for t in 1:Tmax
        # apply recoveries
        for i in 1:N
            if current_state[i] == 1 && t == recover_time[i]
                current_state[i] = -1
            end
        end

        θ = theta2 !== nothing ? (t <= t_star ? theta1 : theta2) : theta1
        infected_count = count(==(1), current_state)  
        
    

        for i in 1:N
            if current_state[i] == 0
                p_inf = 1 - (1 - θ/N)^infected_count
              
                if rand() < p_inf
                    w_ik    = t
                    tau_ikp = rand(truncated(Normal(4.2, 0.5), 1e-3, Inf))
                    extra   = rand(truncated(Normal(7.3, 0.6), 1e-3, Inf))
                    tau_ikr = tau_ikp + extra
                    delta_ik= rand(Exponential(20.0))
                    v_ik    = w_ik + tau_ikp

                    push!(history[i], (w_ik=w_ik, tau_ikp=tau_ikp, tau_ikr=tau_ikr,
                                       delta_ik=delta_ik, v_ik=v_ik))

                    recover_time[i] = t + ceil(Int, tau_ikr)
                    current_state[i] = 1
                end
            end
        end

        if t < Tmax
            states[:, t+1] .= current_state
        end
    end

    return (states=states, history=history)
end


"""
    posterior_summary(posterior; true_t_star=nothing, tolerance=2)

Summarize the changepoint MCMC output from `changepoint_mcmc`.
"""
function posterior_summary(
    posterior;
    true_t_star::Union{Nothing,Int}=nothing,
    tolerance::Int=2,
)
    samples = posterior.samples

    if size(samples, 1) == 0
        return (
            theta1_median = missing,
            theta2_median = missing,
            theta_diff_median = missing,
            tau_median = missing,
            tau_map = missing,
            tau_q025 = missing,
            tau_q975 = missing,
            p_positive_effect = missing,
            p_tau_near_truth = missing,
            accepted_rate = posterior.accepted_rate,
        )
    end

    theta1 = samples[:, 1]
    theta2 = samples[:, 2]
    tau = Int.(round.(samples[:, 3]))
    theta_diff = theta2 .- theta1

    p_tau_near_truth = true_t_star === nothing ? missing :
        mean(abs.(tau .- true_t_star) .<= tolerance)

    return (
        theta1_median = median(theta1),
        theta2_median = median(theta2),
        theta_diff_median = median(theta_diff),
        tau_median = Int(round(median(tau))),
        tau_map = mode(tau),
        tau_q025 = quantile(tau, 0.025),
        tau_q975 = quantile(tau, 0.975),
        p_positive_effect = mean(theta_diff .> 0),
        p_tau_near_truth = p_tau_near_truth,
        accepted_rate = posterior.accepted_rate,
    )
end

"""
    is_detected(summary, analysis_day; ...)

Decision rule for online detection.

The current MCMC assumes a changepoint exists, so we should not declare a
change just because it returns a tau. This rule declares a change when:

    Pr(theta2 > theta1 | data) > p_effect_threshold

and the posterior median changepoint is at least `guard_days` before the
current analysis day.
"""
function is_detected(
    summary,
    analysis_day::Int;
    p_effect_threshold::Float64=0.70,
    guard_days::Int=4,
)
    if ismissing(summary.p_positive_effect) || ismissing(summary.tau_median)
        return false
    end

    return summary.p_positive_effect > p_effect_threshold &&
           summary.tau_median <= analysis_day - guard_days
end

"""
    online_changepoint_detection(X_obs; ...)

Run the existing changepoint MCMC repeatedly as data accumulate. For example,
with `breaks=2` and `analysis_every=10`, analyses are run after every five
dust observations.
"""
function online_changepoint_detection(
    X_obs::AbstractVector{<:Integer};
    N::Int,
    Tmax::Int,
    xi::Float64=log(2) / 7,
    breaks::Int=2,
    analysis_every::Int=10,
    min_analysis_day::Int=20,
    true_t_star::Union{Nothing,Int}=nothing,
    tolerance::Int=2,
    n_iter::Int=3000,
    burn_in::Int=500,
    thin::Int=1,
    prop_sd::NamedTuple=(theta1=0.05, theta2=0.05, tau=2),
    p_effect_threshold::Float64=0.70,
    guard_days::Int=4,
    stop_after_detection::Bool=false,
    track_timing::Bool=true,
    verbose::Bool=false,
)
    analysis_days = collect(min_analysis_day:analysis_every:Tmax)
    records = DataFrame()
    first_detection_day = missing
    first_detection_tau = missing

    for analysis_day in analysis_days
        n_obs = div(analysis_day, breaks)
        if n_obs < 2 || n_obs > length(X_obs)
            continue
        end

        if track_timing
            elapsed = @elapsed posterior = changepoint_mcmc(
                collect(X_obs[1:n_obs]);
                N = N,
                Tmax = analysis_day,
                xi = xi,
                breaks = breaks,
                n_iter = n_iter,
                burn_in = burn_in,
                thin = thin,
                prop_sd = prop_sd,
            )
        else
            elapsed = missing
            posterior = changepoint_mcmc(
                collect(X_obs[1:n_obs]);
                N = N,
                Tmax = analysis_day,
                xi = xi,
                breaks = breaks,
                n_iter = n_iter,
                burn_in = burn_in,
                thin = thin,
                prop_sd = prop_sd,
            )
        end

        summary = posterior_summary(
            posterior;
            true_t_star = true_t_star,
            tolerance = tolerance,
        )

        detected = is_detected(
            summary,
            analysis_day;
            p_effect_threshold = p_effect_threshold,
            guard_days = guard_days,
        )

        if detected && ismissing(first_detection_day)
            first_detection_day = analysis_day
            first_detection_tau = summary.tau_median
        end

        push!(records, (
            analysis_day = analysis_day,
            n_observations = n_obs,
            detected = detected,
            elapsed_seconds = elapsed,
            theta1_median = summary.theta1_median,
            theta2_median = summary.theta2_median,
            theta_diff_median = summary.theta_diff_median,
            tau_median = summary.tau_median,
            tau_map = summary.tau_map,
            tau_q025 = summary.tau_q025,
            tau_q975 = summary.tau_q975,
            p_positive_effect = summary.p_positive_effect,
            p_tau_near_truth = summary.p_tau_near_truth,
            accepted_rate = summary.accepted_rate,
        ); cols=:union)

        if verbose
            elapsed_text = ismissing(elapsed) ? "not tracked" : string(round(elapsed; digits=2), "s")
            println(
                "analysis_day=", analysis_day,
                ", detected=", detected,
                ", tau_median=", summary.tau_median,
                ", theta_diff_median=", round(summary.theta_diff_median; digits=3),
                ", elapsed=", elapsed_text,
            )
        end

        if detected && stop_after_detection
            break
        end
    end

    return (
        records = records,
        first_detection_day = first_detection_day,
        first_detection_tau = first_detection_tau,
    )
end

"""
    run_one_replicate(rep; ...)

Simulate one full outbreak/dust trajectory, then apply online changepoint
detection to prefixes of the observed dust time series.
"""
function run_one_replicate(
    rep::Int;
    seed::Int=2026,
    N::Int=1000,
    Tmax::Int=60,
    theta1::Float64=0.15,
    theta2::Float64=0.25,
    true_t_star::Int=45,
    xi::Float64=log(2) / 7,
    breaks::Int=2,
    analysis_every::Int=10,
    min_analysis_day::Int=20,
    n_iter::Int=3000,
    burn_in::Int=500,
    thin::Int=1,
    prop_sd::NamedTuple=(theta1=0.05, theta2=0.05, tau=2),
    p_effect_threshold::Float64=0.70,
    guard_days::Int=4,
    stop_after_detection::Bool=false,
    track_timing::Bool=true,
    tolerance::Int=2,
    verbose::Bool=false,
)
    Random.seed!(seed + rep - 1)

    if track_timing
        sim_elapsed = @elapsed result = simulate_dust_observation(
            N = N,
            Tmax = Tmax,
            theta1 = theta1,
            theta2 = theta2,
            t_star = true_t_star,
            xi = xi,
            breaks = breaks,
        )
    else
        sim_elapsed = missing
        result = simulate_dust_observation(
            N = N,
            Tmax = Tmax,
            theta1 = theta1,
            theta2 = theta2,
            t_star = true_t_star,
            xi = xi,
            breaks = breaks,
        )
    end

    X_obs = result.observations.Dust

    if track_timing
        detection_elapsed = @elapsed detection = online_changepoint_detection(
            X_obs;
            N = N,
            Tmax = Tmax,
            xi = xi,
            breaks = breaks,
            analysis_every = analysis_every,
            min_analysis_day = min_analysis_day,
            true_t_star = true_t_star,
            tolerance = tolerance,
            n_iter = n_iter,
            burn_in = burn_in,
            thin = thin,
            prop_sd = prop_sd,
            p_effect_threshold = p_effect_threshold,
            guard_days = guard_days,
            stop_after_detection = stop_after_detection,
            track_timing = track_timing,
            verbose = verbose,
        )
    else
        detection_elapsed = missing
        detection = online_changepoint_detection(
            X_obs;
            N = N,
            Tmax = Tmax,
            xi = xi,
            breaks = breaks,
            analysis_every = analysis_every,
            min_analysis_day = min_analysis_day,
            true_t_star = true_t_star,
            tolerance = tolerance,
            n_iter = n_iter,
            burn_in = burn_in,
            thin = thin,
            prop_sd = prop_sd,
            p_effect_threshold = p_effect_threshold,
            guard_days = guard_days,
            stop_after_detection = stop_after_detection,
            track_timing = track_timing,
            verbose = verbose,
        )
    end

    interim = detection.records
    interim.rep = fill(rep, nrow(interim))
    interim.seed = fill(seed + rep - 1, nrow(interim))
    interim.true_theta1 = fill(theta1, nrow(interim))
    interim.true_theta2 = fill(theta2, nrow(interim))
    interim.true_t_star = fill(true_t_star, nrow(interim))

    detected = !ismissing(detection.first_detection_day)
    detection_delay = detected ? detection.first_detection_day - true_t_star : missing
    false_alarm = detected && detection.first_detection_day < true_t_star

    summary = DataFrame([(
        rep = rep,
        seed = seed + rep - 1,
        true_theta1 = theta1,
        true_theta2 = theta2,
        true_t_star = true_t_star,
        detected = detected,
        first_detection_day = detection.first_detection_day,
        first_detection_tau = detection.first_detection_tau,
        detection_delay = detection_delay,
        false_alarm = false_alarm,
        sim_elapsed_seconds = sim_elapsed,
        detection_elapsed_seconds = detection_elapsed,
        total_elapsed_seconds = track_timing ? sim_elapsed + detection_elapsed : missing,
    )])

    return (summary = summary, interim = interim, observations = result.observations)
end

function performance_summary(
    summaries::DataFrame;
    n_reps::Int,
    theta1::Float64,
    theta2::Float64,
    true_t_star::Int,
    p_effect_threshold::Float64,
    guard_days::Int,
    analysis_every::Int,
    min_analysis_day::Int,
    stop_after_detection::Bool,
    track_timing::Bool,
    started_at::DateTime,
    total_elapsed::Float64,
    completed::Bool,
)
    detection_rate = nrow(summaries) == 0 ? missing : mean(summaries.detected)
    false_alarm_rate = nrow(summaries) == 0 ? missing : mean(summaries.false_alarm)
    detected_delays = nrow(summaries) == 0 ? [] : collect(skipmissing(summaries.detection_delay))
    median_delay = isempty(detected_delays) ? missing : median(detected_delays)
    completed_reps = nrow(summaries)

    return DataFrame([(
        n_reps = n_reps,
        completed_reps = completed_reps,
        completed = completed,
        true_theta1 = theta1,
        true_theta2 = theta2,
        true_t_star = true_t_star,
        p_effect_threshold = p_effect_threshold,
        guard_days = guard_days,
        analysis_every = analysis_every,
        min_analysis_day = min_analysis_day,
        stop_after_detection = stop_after_detection,
        track_timing = track_timing,
        detection_rate = detection_rate,
        false_alarm_rate = false_alarm_rate,
        median_detection_delay = median_delay,
        started_at = started_at,
        finished_at = now(),
        total_elapsed_seconds = total_elapsed,
        avg_elapsed_seconds_per_rep = completed_reps == 0 ? missing : total_elapsed / completed_reps,
    )])
end

"""
    simulation_study(; n_reps=50, theta1=0.15, theta2=0.25, true_t_star=45, ...)

Run a simulation study and return summary and interim-analysis DataFrames.
Use `theta1`, `theta2`, and `true_t_star` to define the data-generating
scenario. 
"""
function simulation_study(;
    n_reps::Int=50,
    seed::Int=2026,
    N::Int=1000,
    Tmax::Int=60,
    theta1::Float64=0.15,
    theta2::Float64=0.25,
    true_t_star::Int=45,
    xi::Float64=log(2) / 7,
    breaks::Int=2,
    analysis_every::Int=10,
    min_analysis_day::Int=20,
    n_iter::Int=3000,
    burn_in::Int=500,
    thin::Int=1,
    prop_sd::NamedTuple=(theta1=0.05, theta2=0.05, tau=2),
    p_effect_threshold::Float64=0.70,
    guard_days::Int=4,
    stop_after_detection::Bool=false,
    track_timing::Bool=true,
    tolerance::Int=2,
    verbose::Bool=true,
)
  
    started_at = now()
    start_seconds = time()
    summaries = DataFrame()
    interims = DataFrame()
    completed = false

    try
        for rep in 1:n_reps
            if verbose
                println("Starting replicate ", rep, "/", n_reps, " at ", Dates.format(now(), "HH:MM:SS"))
            end

            result = run_one_replicate(
                rep;
                seed = seed,
                N = N,
                Tmax = Tmax,
                theta1 = theta1,
                theta2 = theta2,
                true_t_star = true_t_star,
                xi = xi,
                breaks = breaks,
                analysis_every = analysis_every,
                min_analysis_day = min_analysis_day,
                n_iter = n_iter,
                burn_in = burn_in,
                thin = thin,
                prop_sd = prop_sd,
                p_effect_threshold = p_effect_threshold,
                guard_days = guard_days,
                stop_after_detection = stop_after_detection,
                track_timing = track_timing,
                tolerance = tolerance,
                verbose = false,
            )

            append!(summaries, result.summary; cols=:union)
            append!(interims, result.interim; cols=:union)

            if verbose
                first_day = result.summary.first_detection_day[1]
                first_day_text = ismissing(first_day) ? "missing" : string(first_day)
                println(
                    "Finished replicate ", rep,
                    ". detected=", result.summary.detected[1],
                    ", first_detection_day=", first_day_text,
                )
            end
        end

        completed = true
    catch err
        if err isa InterruptException
            println()
            println("Simulation interrupted. Returning partial results for ", nrow(summaries), " completed replicates.")
        else
            rethrow(err)
        end
    end

    total_elapsed = time() - start_seconds
    performance = performance_summary(
        summaries;
        n_reps = n_reps,
        theta1 = theta1,
        theta2 = theta2,
        true_t_star = true_t_star,
        p_effect_threshold = p_effect_threshold,
        guard_days = guard_days,
        analysis_every = analysis_every,
        min_analysis_day = min_analysis_day,
        stop_after_detection = stop_after_detection,
        track_timing = track_timing,
        started_at = started_at,
        total_elapsed = total_elapsed,
        completed = completed,
    )

    return (
        performance = performance,
        summaries = summaries,
        interims = interims,
    )
end

"""
    scenario_grid(; theta1_values=[0.1], theta2_values=[0.15, 0.2, 0.25],
                    true_t_star_values=[15, 30, 45], ...)

Run `simulation_study` over every combination of the supplied theta and
changepoint settings. Extra keyword arguments are passed to `simulation_study`.
"""
function scenario_grid(;
    theta1_values::AbstractVector{<:Real}=[0.1],
    theta2_values::AbstractVector{<:Real}=[0.15, 0.2, 0.25],
    true_t_star_values::AbstractVector{<:Integer}=[15, 30, 45],
    kwargs...,
)
    performances = DataFrame()
    summaries = DataFrame()
    interims = DataFrame()

    for theta1 in theta1_values
        for theta2 in theta2_values
            for true_t_star in true_t_star_values
                println(
                    "Scenario: theta1=", theta1,
                    ", theta2=", theta2,
                    ", true_t_star=", true_t_star,
                )

                result = simulation_study(;
                    theta1 = Float64(theta1),
                    theta2 = Float64(theta2),
                    true_t_star = Int(true_t_star),
                    kwargs...,
                )

                append!(performances, result.performance; cols=:union)
                append!(summaries, result.summaries; cols=:union)
                append!(interims, result.interims; cols=:union)
            end
        end
    end

    return (
        performance = performances,
        summaries = summaries,
        interims = interims,
    )
end

"""
    save_results_csv(results; output_dir=joinpath(@__DIR__, "..", "results"), prefix="simulation_results")

Save the three DataFrames returned by `simulation_study` or `scenario_grid`
to CSV files:

  - `<prefix>_performance.csv`
  - `<prefix>_summaries.csv`
  - `<prefix>_interims.csv`
"""
function save_results_csv(
    results;
    output_dir::AbstractString=joinpath(@__DIR__, "..", "results"),
    prefix::AbstractString="simulation_results",
)
    try
        @eval import CSV
    catch err
        error("CSV.jl is required to save results. Run `using Pkg; Pkg.add(\"CSV\")`, then try again.")
    end

    output_dir = abspath(output_dir)
    mkpath(output_dir)

    paths = (
        performance = joinpath(output_dir, string(prefix, "_performance.csv")),
        summaries = joinpath(output_dir, string(prefix, "_summaries.csv")),
        interims = joinpath(output_dir, string(prefix, "_interims.csv")),
    )

    CSV.write(paths.performance, results.performance)
    CSV.write(paths.summaries, results.summaries)
    CSV.write(paths.interims, results.interims)

    println("Saved CSV files:")
    println("  ", paths.performance)
    println("  ", paths.summaries)
    println("  ", paths.interims)

    return paths
end
