# AI Forecasting Hackathon Submission
# Simulating Automation Timelines Through Labor-Capability Modeling  
*A probabilistic AI-labor forecasting system with a focus on India and Nigeria*  

### Authors  
- **Arya Hariharan** – Independent  
- **Marie-Louise Thurton** – Toronto Metropolitan University  
- **Oluwagbemike Olowe** – Data Engineer, Independent  
- **Ifeoma Ilechukwu** – AI Legal & Governance Advisor, Independent  
- **Rijal Saepuloh** – Independent  

With **Apart Research**  
*November 2, 2025*  


## Overview  

**Simulating Automation Timelines Through Labor-Capability Modeling** is a forecasting system designed to estimate how quickly AI will automate various occupations.  
The model integrates **AI capability benchmarks**, **labor skill data**, and **policy indicators** to produce **probabilistic forecasts** through 2035.  

## Objectives  

- Model **automation vulnerability** of 894 occupations based on their skill composition.  
- Incorporate **AI capability growth trajectories** using time-series benchmark data.  
- Simulate **policy and regional constraints** (e.g., regulation, infrastructure, investment).  
- Generate a live, **policy-aware “Forecast of Work” dashboard** for workforce planning.  


## Data Sources  

| Category | Description | Source |
|-----------|--------------|--------|
| **Occupational Skills** | 894 occupations with 35 core skill requirements (importance and level ratings). | O*NET 30.0 |
| **AI Capabilities** | Time-series benchmark scores for 5 key domains: MATH, SWE-bench, MedQA, MMLU-Pro, GPQA. | LLM Stats, OpenAI/Anthropic benchmark datasets |
| **Labor Distribution** | Employment shares by occupation. | PLFS (India), NBS (Nigeria) |
| **Regional Indicators** | GDP per capita, R&D expenditure, electricity access, governance quality. | World Bank WDI & WGI |
| **Policy Events** | Real-time AI regulation and investment signals. | GDELT |

## References  

The model leverages benchmark datasets and prior works including:  
- Hendrycks et al. (2021) – *MATH Dataset*  
- Jimenez et al. (2023) – *SWE-Bench*  
- Jin et al. (2020) – *MedQA*  
- Wang et al. (2024) – *MMLU-Pro*  
- Rein et al. (2023) – *GPQA*  
- Srivastava et al. (2022) – *Beyond the Imitation Game*  

Full reference list included in the accompanying paper.  

## Citation  

If you use or reference this work, please cite:  

> Hariharan, A., Thurton, M.L., Olowe, O., Ilechukwu, I., & Saepuloh, R. (2025).  
> *Simulating Automation Timelines Through Labor-Capability Modeling.*  
> Apart Research – Track 1: Regional Constraints, November 2025.  
