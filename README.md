# Final Work - Thermodynamic Relations and Phase Equilibrium - Chemical Process Engineering Specialization Course

<br>

## I. Project Description

<br>

This work aims to design a power-refrigeration combined thermodynamic cycle. All the work produced in the turbine from power cycle will be used to feed the pump of the power cycle and also the compression unit of the refrigeration cycle. The operational points of both cycles must be selected by the engineering team, but they must be enough to reach project's specifications.

The Python implemention aims to help the fast evaluation of thermodynamic properties and to permit fast investigations of different operational setups. On the backend, both classical thermodynamic models and machine learning approaches are used to estimate thermodynamic properties.

<br>

## II. General Objectives

<br>

To design a combined thermodynamic cycle and to evaluate the thermodynamic performance from each individual cycle, through an energetic and an exertic analysis. These analysis must indicate options to enhance performance. The combined cycle must mantain a refrigeration capacity of 1,500 RT (refrigeration tons) in the evaporator unit - 1 RT = 12,000 BTU/h

<br>

## III. Project Specifications

<br>

### 3.1. Refrigeration Cycle

<br>

The refrigeration cycle is a conventional cycle of vapor compression and expasion, with the four basic units: compression, condenser, isoenthalpic expansion valve and evaporator. O cycle must offer a refrigeration rate of 1,500 RT at the evapaorator unit. This "cold demand" is used to cool a water stream at 100 kPa, from 60 °C to 15 °C, which will be used as cold utility at distillation unit condensers.

The available refrigerants are ammonia (R717) or methane (R50). The temperatures and pressures from each stream are selected by engineering teams.

#### 3.1.1. Important hypotheses:

<ol>
    <li>The compression unit does not operate reversibly, and it has an isoentropic efficiency (defined by engineering team). Also, it does not operate adiabatically. The heat loss is estimatet to be 5 % of the net demanded power.</li>
    <li>The condenser exiting stream is assumed to be at saturated liquid state</li>
    <li>The evaporator exiting stream is assumed to be at saturated vapor state</li>
    <li>The condenser uses cold water available at 20 °C and the maximum exiting temperature of such water is 60 °C</li>
    <li>All other units from the cycle (condenser, evaporator and expansion valve) operate adiabatically</li>
    <li>The pipelines that connect the equipments are assumed as adiabatic and the head loss (due to fluid friction) can be neglected</li>
</ol>

<br>

### 3.2. Power Cycle

<br>

The power cycle is a conventional Rankine cycle, with the four main units (boiler, turbine, condenser and pump). It uses water as work fluid. The temperatures and pressures from each stream are selected by engineering teams.

#### 3.2.1. Important hypotheses

<ol>
    <li>Turbine operates in two stages and they are not reversible. The isoentropic efficiency is defined by engineering team</li>
    <li>The stages are not adiabatic, and the heat loss is estimated as 5 % of gross power produced</li>
    <li>Between the turbine stages, a fraction of steam stream is redirected to a methanol plant (this purge is at the same conditions of the exiting strem from stage 1)</li>
    <li>The boiler does not operate adiabatically. The heat loss is estimated as 10 % of the necessary heat given to work fluid</li>
    <li>A fraction of obtained work in the turbine is used to power the pump</li>
    <li>The pump does not operate reversibly, and the isoentropic efficiency must be designed by the team</li>
    <li>The pump and the condenser can be assumed adiabatic</li>
    <li>The condenser exiting stream is at saturated liquid stream</li>
    <li>The make-up stream of water is at the same conditions of the condenser exiting stream</li>
    <li>The boiler uses a hot gases stream, coming from a combustion process, as heat source. These gases can be admitted as having the same properties of air at ideal gas state. This stream is available at 100 kPa and 950 K</li>
    <li>The condenser uses cold water available at 20 °C and the maximum exiting temperature of such water is 60 °C</li>
</ol>