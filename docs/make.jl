using GKWExperiments
using Documenter

DocMeta.setdocmeta!(GKWExperiments, :DocTestSetup, :(using GKWExperiments); recursive=true)

makedocs(;
    modules=[GKWExperiments],
    authors="Isaia Nisoli",
    sitename="GKWExperiments.jl",
    format=Documenter.HTML(;
        canonical="https://orkolorko.github.io/GKWExperiments.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/orkolorko/GKWExperiments.jl",
    devbranch="main",
)
