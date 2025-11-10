## Import some useful packages

# Used for plotting bifurcation diagrams and ODE solutions
using BifurcationKit
using DifferentialEquations
using ForwardDiff
using Measures
using Plots
using LaTeXStrings
using ColorSchemes

# Needed for interfacing with Python package
using PyCall

# Import our python package with PyCall
asp = pyimport("auxin_signalling_pathway")

output_dir = "output/bifurcation_example"

mkpath(output_dir)

# Generate a "reduced" model with 1 ARF and 1 IAA. This model uses a QSS assumption for promoter
# binding and unbinding

# MP is denoted ARF_1 and BDL denoted iaa_1. You could change this if you really wanted to

asp_model = asp.reduced_auxin_signalling_pathway(nARFs=1, nIAAs=1,
                                                 include_arf_transcription=true)

# Print out default parameter labels
param_dict = asp_model.get_default_parameters()

param_labels = sort(collect(keys(param_dict)))
println(param_labels)

# Get the "substitution" dictionary which can be used to modify the parameterisation
parameter_subs_dict = asp_model.get_param_substitution_dict()

state_labels = asp_model.get_state_variables()

println("states are $(state_labels)")

# Create dictionary of parameters to change
new_params = Dict()
new_params["k__D__ARF_1__ARF_1"] = 5e-2
new_params["d__D__ARF_1__ARF_1"] = 1e-2

new_params["k__D__ARF_1__iaa_1"] = 5e-2
new_params["d__D__ARF_1__iaa_1"] = 1e-2

new_params["k__D__iaa_1__iaa_1"] = 5e-2
new_params["d__D__iaa_1__iaa_1"] = 1e-2

new_params["d__R"] = 1e-1
new_params["k__RNA__IAA"] = 1.0
new_params["k__RNA__ARF"] = 1.0

new_params["d__iaa_1__x"] = 1.0

new_params["k__G__ARF_1__ARF_1__R__iaa_1"] = 1.0

new_params["d__arf__all"] = 2.0
new_params["d__iaa__all"] = 0.1

new_params["k__G__ARF_1__iaa_1__R__ARF_1"] = 0.0
new_params["k__G__ARF_1__ARF_1__R__ARF_1"] = 1.0
new_params["k__G__R__ARF_1"] = 0.0001

new_params["k__G__ARF_1__ARF_1__R__iaa_1"] = 1.0
new_params["k__G__ARF_1__iaa_1__R__iaa_1"] = 0.0
new_params["k__G__R__iaa_1"] = 0.0001


# Insert new parameters, ensuring that all are present in the model

# Only include parameters that already exist in the model
for key in keys(new_params)
    if key in keys(param_dict)
        param_dict[key] = new_params[key]
    else
        println("WARNING param $(key) is not in param dict")
    end
end

println(param_dict)

# Generate derivative function using the Python package
f_deriv_func = asp_model.output_julia()
f_deriv = eval(Meta.parse(f_deriv_func))


## Set up a dictionary of max values to use in plots
# Set all max_p values to 1.0 by default
max_p_dict = Dict()
for (k, v) in param_dict
    max_p_dict[k] = 1.0
end

max_p_dict["k__arf__all"] = 1.0
max_p_dict["d__arf__all"] = 1.0

max_p_dict["d__iaa__all"] = 0.5
max_p_dict["k__D__ARF_1__ARF_1"] = 1e1
max_p_dict["k__D__ARF_1__iaa_1"] = 1e1
max_p_dict["k__D__iaa_1__iaa_1"] = 1e1
max_p_dict["k__RNA__IAA"] = 1e1
max_p_dict["d__R"] = 1e1
max_p_dict["d__D__ARF_1__iaa_1"] = 1e1
max_p_dict["auxin"] = 10.0

par_pp = Float64.(collect(values(sort(param_dict))))
push!(par_pp, 10.0)
println(par_pp)

# Solve system with high and low auxin
tspan = (0.0, 10000.0)
z0 = Float64.(collect(values(sort(asp_model.get_default_initial_conditions()))))
println(z0)
println(param_dict)
println(par_pp)

prob = ODEProblem(f_deriv, z0, tspan, par_pp)
sol = DifferentialEquations.solve(prob)
z0 = sol[:, size(sol, 2)]

s = plot(sol)
savefig(s, "$(output_dir)/ode_sol.pdf")

print("SS with auxin conc. = 10.0, ")
println(Dict(zip(state_labels, z0)))

legend_labels = reshape(state_labels, 1, length(state_labels))

xlabel = "d__iaa__all"
param_index = findfirst(==(xlabel), param_labels)

const default_ic_dict = asp_model.get_default_initial_conditions()
function iaa_total_func(x)
    include_indices = []
    for (i, key) in enumerate(state_labels)
        if occursin("iaa", key) && !occursin("R__", key) && !occursin("G__", key)
            for j in 1:length(collect(eachmatch(Regex("iaa"), key)))
                push!(include_indices, i)
            end
        end
    end
    return sum(x[include_indices])
end

const default_ic_dict = asp_model.get_default_initial_conditions()
function arf_total_func(x)
    include_indices = []
    for (i, key) in enumerate(state_labels)
        if occursin("ARF", key) && !occursin("R__", key) && !occursin("G__", key)
            for j in 1:length(collect(eachmatch(Regex("ARF"), key)))
                push!(include_indices, i)
            end
        end
    end
    return sum(x[include_indices])
end

function iaa_rna_total_func(x)
    include_indices = []
    for (i, key) in enumerate(state_labels)
        if startswith(key, "R__iaa")
            push!(include_indices, i)
        end
    end
    return sum(x[include_indices])
end


function arf_rna_func(x, arf_index=nothing)
    arfs = asp_model.arf_symbols
    ret_vec = [0.0 for a in arfs]

    if arf_index === nothing
      for (i, arf) in enumerate(arfs)
          state_index = findfirst(==("R__$(arf)"), state_labels)
          ret_vec[i] = x[state_index]
      end
        return ret_vec
    end

    arf = arfs[arf_index]
    rna_symbol = "R__$(arf)"
    state_index = findfirst(==(rna_symbol),
                            state_labels)
    return x[state_index]
end

recordFromSolution(x, p; k...) = (rna_iaa_tot=iaa_rna_total_func(x), iaa_tot=iaa_total_func(x),
                                  arf1_rna=arf_rna_func(x, 1), arf_tot=arf_total_func(x))

# z0_2 = copy(Float64.(collect(values(sort(asp_model.get_default_initial_conditions())))))
par_pp[length(par_pp)] = 0.0
par_pp[param_index] = 0.0349

z0_2 = [0.003942030760178107, 7.769803257093408e-5,
        0.011156888791130574, 1.6020504429472349, 0.007884061520356214,
        0.007884061520356214, 0.5660477794265649]

prob = ODEProblem(f_deriv, z0_2, tspan, par_pp)
sol = DifferentialEquations.solve(prob)
z0_2 = sol[:, size(sol, 2)]

println(Dict(zip(state_labels, z0_2)))
par_pp[length(par_pp)] = 0.0

prob = ODEProblem(f_deriv, z0_2, tspan, par_pp)
sol = DifferentialEquations.solve(prob)

z0_2 = sol[:, size(sol, 2)]

println(Dict(zip(state_labels, z0_2)))

par_pp[length(par_pp)] = 0.0

println(Dict(zip(state_labels, z0_2)))
par_pp[length(par_pp)] = 0.0

# bifurcation problem
prob1 = BifurcationProblem(f_deriv, z0, par_pp,
                           # specify the continuation parameters
                           (@optic _[param_index]),
                           record_from_solution=recordFromSolution,
                           )


prob2 = BifurcationProblem(f_deriv, z0_2, par_pp,
                           # specify the continuation parameters
                           (@optic _[param_index]),
                           record_from_solution=recordFromSolution,
                           )

p_max = max_p_dict[xlabel]
p_init = p_max / 10

dsmax = p_max / 10
dsmin = p_init * 1e-8

ds = (dsmax + dsmin) / 2.0

opts_br = ContinuationPar(p_max = p_max, p_min=1e-6,
                          dsmax=dsmax, dsmin=dsmin, ds = ds,
                          nev=10, max_steps=2000,
                          detect_bifurcation=3)

opts_br2 = ContinuationPar(p_max = p_max, p_min=1e-3,
                          dsmax=dsmax, dsmin=dsmin, ds = ds,
                          nev=10, max_steps=2000,
                          detect_bifurcation=3)

diagram1 = bifurcationdiagram(prob1, PALC(),
                              3,
                              opts_br,
                              bothside=true
                              )

diagram2 = bifurcationdiagram(prob2, PALC(),
                              3,
                              opts_br2,
                              bothside=true
                              )

println(diagram1)
println(diagram2)

scene = plot(diagram1; code=(), legend=true,
             vars=(:param, :iaa_tot))

dpi = 100
size_inches = (4.25, 3.5)
size_px = Tuple(round.(Int, dpi .* size_inches))

branch = diagram1.γ

fieldnames(typeof(branch))
branch.specialpoint

hopf_points = [p for p in branch.specialpoint if string(p.type)=="hopf"]

hopf_point = hopf_points[1]

start_step = hopf_point.step + 10
end_step = hopf_point.step - 10

x1 = branch.sol[findfirst(x->x.step == start_step, branch.sol) + 1]
x2 = branch.sol[argmin([abs(x.p - x1.p) for x in [x for x in branch.sol if sum((x1.x - x.x).^2) > 100]])]

println(x1, x2)

par_pp[param_index] = x1.p


plot(diagram1, vars=(:param, :iaa_tot))
c_start = get(ColorSchemes.viridis, 1.0)
c_end   = get(ColorSchemes.viridis, 0.0 )

param_val = par_pp[param_index]

scatter!([param_val, param_val], [iaa_total_func(x1.x), iaa_total_func(x2.x)],
         markershape = :x,
         markerstrokewidth=1.0,
         color = [c_start, c_end],
         legend = false,
)

plot!([x1.p, x1.p],
      [iaa_total_func(x1.x), iaa_total_func(x2.x)], linestyle=:dash,
      color=:grey)

p1 = plot!(diagram2; code=(),
           vars=(:param, :iaa_tot),
           legend=false,
           ylabel="IAA conc. total",
           xlabel=L"d_I",
           guidefontsize=9,
           labelfontsize=9
           )

N = 250
ic_vec = [x2.x .+ t .* (x1.x .- x2.x) for t in range(0, 1, length=N)]
p_vec = [x1.p for t in range(0, 1, length=N)]

colors = range(0, 1, length=length(ic_vec))  # Normalize to [0,1]
cmap = cgrad(:viridis, length(ic_vec))

p2 = plot(ylabel="IAA conc. total", xlabel=L"t")

for (i, (z0, p)) in enumerate(zip(ic_vec, p_vec))
    local prob
    par_pp[end] = p
    prob = ODEProblem(f_deriv, z0, tspan, par_pp)
    local sol
    sol = DifferentialEquations.solve(prob)
    color = cmap[colors[i]]
    plot!(sol.t, [iaa_total_func(x) for x in sol.u],
    label=false, line_z=iaa_total_func(z0), cmap=:viridis)
end

xticks = ([0, round(Int, maximum(sol.t))], [0, "1"])

p2 = plot!(colorbar=false, xticks=xticks,
guidefontsize=9,
           labelfontsize=9
           )

colors = range(0, 1, length=length(ic_vec))  # Normalize to [0,1]
cmap = cgrad(:viridis, length(ic_vec))

tspan = (0.0, 5e5)

scene = plot(xlabel="ARF conc. total")

xlim_max = 0.0
for (i, z0) in enumerate(ic_vec)
    local prob
    prob = ODEProblem(f_deriv, z0, tspan, par_pp)
    local sol
    sol = DifferentialEquations.solve(prob)
    xvals = [arf_total_func(x) for x in sol.u]
    global xlim_max = max(xlim_max, maximum(xvals))

    yvals = [iaa_total_func(x) for x in sol.u]
    p3 = plot!(xvals, yvals,
               label=false, line_z=iaa_total_func(z0), cmap=:viridis)
end

xticks = [0, round(Int, xlim_max)]
p3 = plot!(colorbar=true, ylabel="", yticks=false,
           xticks=xticks,
           guidefontsize=9,
           labelfontsize=9
           )

# 'a' is top, 'b' and 'c' are bottom
my_layout = @layout [a; [b c]]

annotate!(p1, (0, 250, text(L"\mathbf{a}", 12, :black, :left)))
annotate!(p2, (0, 500, text(L"\mathbf{b}", 12, :black, :left)))
annotate!(p3, (0, 500, text(L"\mathbf{c}", 12, :black, :left)))

plot(p1, p2, p3; layout=my_layout,
     labelfontsize=9,
     guidefontsize=9,
     size=size_px,
     leftmargin=2mm,
     dpi=dpi,
     xgrid=false,
     ygrid=false)

savefig("$(output_dir)/bifurcation_diagram.pdf")
