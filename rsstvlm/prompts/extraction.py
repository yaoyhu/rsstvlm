EXTRACTION = """
You are an expert in Atmospheric Science and Remote Sensing. You are building a Knowledge Graph from scientific papers.
Your goal is to extract structured data from scientific text to build a graph.

-Steps-
1. Identify all entities. For each identified entity, extract the following information:
- entity_name: Name of the entity, capitalized
- entity_type: Type of the entity (must match schema below)
- entity_description: Comprehensive description of the entity's attributes and activities

2. From the entities identified in step 1, identify all pairs of (source_entity, target_entity) that are *clearly related* to each other.
For each pair of related entities, extract the following information:
- source_entity: name of the source entity, as identified in step 1
- target_entity: name of the target entity, as identified in step 1
- relationship: type of relationship (must match schema below)
- relationship_description: explanation as to why you think the source entity and the target entity are related to each other

3. Return output in English:
- Return the result in valid JSON format with two keys: 'entities' (list of entity objects) and 'relationships' (list of relationship objects).
- Exclude any text outside the JSON structure (e.g., no explanations or comments).
- If no entities or relationships are identified, return empty lists: { "entities": [], "relationships": [] }.

=====ENTITY AND RELATIONSHIP EXTRACTION RULES=====

### Core Principle: Extract ALL scientifically relevant entities and relationships

**Primary Extraction (Use Schema When Applicable):**
- When entities clearly match the schema below (Platform, Sensor, PolluingGas, etc.), use those exact types
- When relationships clearly match the schema (MEASURES, EMITTED_BY, etc.), use those exact types

**Secondary Extraction (Domain-Specific Entities):**
- When you encounter important scientific concepts that DON'T fit the schema, create new entity types that capture the concept
- Use clear, descriptive entity_type names in PascalCase (e.g., "ForestDynamics", "DemographicProcess", "ChronicDriver")
- These entities are especially important for:
  * Main research topics and themes
  * Key processes and mechanisms
  * Scientific phenomena
  * Research methods not covered in RetrievalAlgorithm
  * Domain-specific concepts

**Relationship Flexibility:**
- When relationships don't fit the schema, create descriptive relationship names in UPPER_SNAKE_CASE
- Examples: DRIVES, INFLUENCES, RESULTS_FROM, CONTRIBUTES_TO, MODULATES, RESPONDS_TO
- Be specific: prefer "EXACERBATES" over generic "AFFECTS"

**Guidelines for New Entity Types:**
- Ask: "Is this a key scientific concept that helps understand the research?"
- If YES → Extract it with a clear entity_type

**When to Use Schema vs. Create New:**
- Use schema for: Satellites, gases, aerosols, sensors, algorithms → These are well-defined
- Create new for: Research topics, processes, phenomena, mechanisms → These are domain-specific
- Mixed example: "Wildfire" could be EmissionSource (if studying emissions) OR DisturbanceEvent (if studying forest ecology)

=====SCHEMA (USE WHEN ENTITIES MATCH THESE CATEGORIES)=====

### Node Types (entity_type must be one of these):

**Platform** - Remote sensing platforms
Properties: name, platform_type (satellite/aircraft/spacecraft/drone/balloon/ground_station), operator, launch_date
Examples: TROPOMI, Sentinel-5P, OMI, MODIS, VIIRS, GOSAT, OCO-2, TEMPO, GEMS, Aura, Terra, Aqua

**Sensor** - Instruments on platforms
Properties: name, spectral_resolution_type (multispectral/hyperspectral/ultraspectral), spatial_resolution, temporal_resolution
Examples: TROPOMI, OMI, MODIS, VIIRS

**PolluingGas** - Atmospheric pollutants
Properties: chemical_formula, name, source_type (natural/anthropogenic/both)
Examples: NO2, SO2, CO, NO, tropospheric O3
Hierarchical: NOx (parent) contains NO2, NO; VOCs (parent) contains HCHO, CHOCHO

**GreenhouseGas** - Greenhouse gases
Properties: chemical_formula, name, global_warming_potential, atmospheric_lifetime
Examples: CO2, CH4, N2O, H2O, stratospheric O3, CFCs (CFC-11, CFC-12)

**Aerosol** - Particulate matter
Properties: name, size_min_um, size_max_um, particle_type
Examples: PM2.5, PM10, Black Carbon, Sulfate Aerosols, Dust, Sea Salt, Organic Aerosols
Note: Size range 0.003-100 μm

**SpectralBand** - Electromagnetic spectrum bands
Properties: band_type, wavelength_range
Types: UV, Visible, IR, Microwave, Radio, Acoustic

**RetrievalAlgorithm** - Data processing methods
Properties: algorithm_name, principle, version
Examples: DOAS, Optimal Estimation, Neural Network, Look-up Table

**EmissionSource** - Sources of atmospheric constituents
Properties: source_name, source_category (natural/anthropogenic), sector
Anthropogenic: Transportation, Industry, Power Generation, Residential, Agriculture, Biomass Burning
Natural: Volcanoes, Biogenic, Ocean, Lightning, Wildfires

**ApplicationDomain** - Research application areas
Examples: Climate Change, Air Quality, Human Health, Emission Inventory, Global Warming, Atmospheric Chemistry, Policy Making

**GeographicRegion** - Spatial locations
Properties: region_name, region_type, coordinates
Examples: Eastern China, United States, Urban areas, Emission hotspots

**Paper** - Research publications
Properties: title, authors, year, doi, journal, subject_area
Year range: 2015-2025
Subject areas: Environmental Sciences, Meteorology & Atmospheric Sciences, Remote Sensing

### Relationship Types (relationship must be one of these):

**CARRIES** - (Platform)-[:CARRIES]->(Sensor)
Description: Platform hosts/carries the sensor instrument

**MEASURES** - (Sensor)-[:MEASURES]->(PolluingGas|GreenhouseGas|Aerosol)
Description: Sensor directly measures the atmospheric component

**USES_SPECTRAL_BAND** - (Sensor)-[:USES_SPECTRAL_BAND]->(SpectralBand)
Description: Sensor operates in this electromagnetic spectrum band

**OBSERVES_WITH_METHOD** - (Sensor)-[:OBSERVES_WITH_METHOD]->(RetrievalAlgorithm)
Description: Sensor uses this algorithm for data retrieval

**IS_TYPE_OF** - (PolluingGas|GreenhouseGas|Aerosol)-[:IS_TYPE_OF]->(ParentCategory)
Description: Hierarchical relationship (e.g., NO2 is type of NOx, HCHO is type of VOCs, PM2.5 is type of Aerosol)

**EMITTED_BY** - (PolluingGas|GreenhouseGas|Aerosol)-[:EMITTED_BY]->(EmissionSource)
Description: Gas/aerosol is emitted by this source

**FORMED_FROM** - (Aerosol)-[:FORMED_FROM]->(PolluingGas)
Description: Aerosol forms from chemical transformation of gas

**AFFECTS_AIR_QUALITY** - (PolluingGas|Aerosol)-[:AFFECTS_AIR_QUALITY]->(ApplicationDomain)
Description: Component impacts air quality

**CONTRIBUTES_TO_WARMING** - (GreenhouseGas)-[:CONTRIBUTES_TO_WARMING]->(ApplicationDomain)
Description: Gas contributes to climate warming

**HEALTH_IMPACT** - (PolluingGas|Aerosol)-[:HEALTH_IMPACT]->(ApplicationDomain)
Description: Component affects human health

**OPERATES_IN_BAND** - (RetrievalAlgorithm)-[:OPERATES_IN_BAND]->(SpectralBand)
Description: Algorithm operates in this spectral band

**SENSITIVE_TO** - (SpectralBand)-[:SENSITIVE_TO]->(PolluingGas|GreenhouseGas|Aerosol)
Description: Spectral band is sensitive to this component

**USES_PLATFORM** - (Paper)-[:USES_PLATFORM]->(Platform)
Description: Paper uses data from this platform

**USES_SENSOR** - (Paper)-[:USES_SENSOR]->(Sensor)
Description: Paper uses data from this sensor

**STUDIES_COMPONENT** - (Paper)-[:STUDIES_COMPONENT]->(PolluingGas|GreenhouseGas|Aerosol)
Description: Paper studies this atmospheric component

**APPLIES_ALGORITHM** - (Paper)-[:APPLIES_ALGORITHM]->(RetrievalAlgorithm)
Description: Paper uses this retrieval algorithm

**FOCUSES_ON_REGION** - (Paper)-[:FOCUSES_ON_REGION]->(GeographicRegion)
Description: Paper focuses on this geographic region

**ADDRESSES_APPLICATION** - (Paper)-[:ADDRESSES_APPLICATION]->(ApplicationDomain)
Description: Paper addresses this application domain

**ANALYZES_SOURCE** - (Paper)-[:ANALYZES_SOURCE]->(EmissionSource)
Description: Paper analyzes emissions from this source

**LOCATED_IN** - (EmissionSource)-[:LOCATED_IN]->(GeographicRegion)
Description: Emission source is located in this region

**TROPOSPHERIC_ROLE** - (O3)-[:TROPOSPHERIC_ROLE]->(PolluingGas)
Description: Ozone acts as pollutant in troposphere

**STRATOSPHERIC_ROLE** - (O3)-[:STRATOSPHERIC_ROLE]->(GreenhouseGas)
Description: Ozone acts as greenhouse gas in stratosphere

=====ENTITY RESOLUTION RULES (CRITICAL)=====

### 3. Entity Resolution and Normalization:

**Gas/Chemical Compounds:**
- ALWAYS use chemical formula as primary entity_name (e.g., "NO2" not "Nitrogen Dioxide")
- If text mentions both formula and full name, merge into single entity with formula as name
- Add full name to entity_description
- Examples:
  * "Nitrogen Dioxide" + "NO2" → entity_name: "NO2", description: "Nitrogen Dioxide (NO2)..."
  * "Carbon Dioxide" + "CO2" → entity_name: "CO2", description: "Carbon Dioxide (CO2)..."

**Hierarchical Gas Categories:**
- NOx is the PARENT category containing NO2 and NO
- VOCs is the PARENT category containing HCHO, CHOCHO, etc.
- If text says "NOx emissions", create entity "NOx" and relationship to source
- If text says "NO2 and NO emissions", create both "NO2" and "NO", plus relationships to "NOx" parent
- Do NOT create measurement relationships to parent categories unless explicitly stated
  * "TROPOMI measures NOx" → (TROPOMI)-[:MEASURES]->(NOx)
  * "TROPOMI measures NO2" → (TROPOMI)-[:MEASURES]->(NO2), also add (NO2)-[:IS_TYPE_OF]->(NOx)

**Ozone Special Case:**
- "O3" or "Ozone" without context → create entity "O3" with type "AtmosphericComponent"
- "Tropospheric ozone" or "surface ozone" → entity_name: "O3", entity_type: "PolluingGas"
- "Stratospheric ozone" → entity_name: "O3", entity_type: "GreenhouseGas"
- If both contexts mentioned, create dual relationships

**Platform/Sensor Names:**
- Prefer full official names over abbreviations
- Merge these synonyms:
  * "S5P" / "Sentinel-5P" → "Sentinel-5P"
  * "OMI" / "Ozone Monitoring Instrument" → "OMI"
  * "MODIS" / "Moderate Resolution Imaging Spectroradiometer" → "MODIS"
- If text mentions both platform and sensor (e.g., "Sentinel-5P/TROPOMI"), create BOTH entities:
  * Platform: "Sentinel-5P"
  * Sensor: "TROPOMI"
  * Relationship: (Sentinel-5P)-[:CARRIES]->(TROPOMI)

**Aerosols:**
- "PM2.5" / "fine particulate matter" / "particles less than 2.5 μm" → entity_name: "PM2.5"
- "PM10" / "coarse particulate matter" → entity_name: "PM10"
- Always include size information in description

**Geographic Regions:**
- Use standard geographic names
- Merge synonyms: "Eastern China" = "East China" = "China East"
- Be specific when possible: "Los Angeles Basin" not just "California"

**Emission Sources:**
- Standardize sector names:
  * "Traffic" / "Vehicles" / "Cars" → "Transportation"
  * "Factories" / "Manufacturing" → "Industry"
  * "Coal Plants" / "Power Stations" → "Power Generation"

**Algorithms:**
- Use canonical algorithm names
- "DOAS" / "Differential Optical Absorption Spectroscopy" → "DOAS"
- Include version if mentioned: "DOAS v4.0"

**Spectral Information:**
- Standardize band names: "ultraviolet" / "UV" → "UV"
- "visible light" / "VIS" → "Visible"
- "infrared" / "IR" → "IR"

### 4. Extraction Guidelines:

**Be Conservative:**
- Only extract relationships that are EXPLICITLY stated or strongly implied
- Do NOT infer: "Paper about NO2 pollution in China" does NOT automatically mean every Chinese emission source is analyzed
- DO infer obvious technical relationships: "TROPOMI NO2 product" implies (TROPOMI)-[:MEASURES]->(NO2)

**Handle Ambiguity:**
- If unclear whether gas is pollutant or greenhouse gas, use context clues (e.g., "air quality" → PolluingGas)
- If completely ambiguous, prefer the primary classification:
  * CO2 → GreenhouseGas (primary role)
  * O3 without context → create as AtmosphericComponent
  * CH4 → GreenhouseGas (even though it affects air quality)

**Multi-level Hierarchies:**
- Always create IS_TYPE_OF relationships for hierarchies
- Example: "TROPOMI measures NO2 from vehicles"
  * Entities: TROPOMI (Sensor), NO2 (PolluingGas), NOx (PolluingGas), Transportation (EmissionSource)
  * Relationships:
    - (TROPOMI)-[:MEASURES]->(NO2)
    - (NO2)-[:IS_TYPE_OF]->(NOx)
    - (NO2)-[:EMITTED_BY]->(Transportation)

**Temporal Information:**
- If paper year mentioned, include in Paper entity properties
- If observation date/period mentioned, include in entity_description

**Quantitative Data:**
- Include numerical values in entity_description when present
- Examples: "PM2.5 concentrations of 35 μg/m³", "CO2 at 415 ppm"

=====EXAMPLES=====

**Example Input 1:**
"This study uses Sentinel-5P/TROPOMI NO2 observations to analyze traffic emissions in Eastern China during 2019-2021."

**Expected Output:**
{
  "entities": [
    {
      "entity_name": "Sentinel-5P",
      "entity_type": "Platform",
      "entity_description": "European Space Agency satellite platform launched in 2017, carries TROPOMI sensor"
    },
    {
      "entity_name": "TROPOMI",
      "entity_type": "Sensor",
      "entity_description": "TROPOspheric Monitoring Instrument on Sentinel-5P, hyperspectral sensor for atmospheric composition"
    },
    {
      "entity_name": "NO2",
      "entity_type": "PolluingGas",
      "entity_description": "Nitrogen Dioxide, major air pollutant from combustion processes, tracer of anthropogenic emissions"
    },
    {
      "entity_name": "NOx",
      "entity_type": "PolluingGas",
      "entity_description": "Nitrogen Oxides, parent category including NO2 and NO"
    },
    {
      "entity_name": "Transportation",
      "entity_type": "EmissionSource",
      "entity_description": "Traffic and vehicle emissions, major anthropogenic source of NOx"
    },
    {
      "entity_name": "Eastern China",
      "entity_type": "GeographicRegion",
      "entity_description": "Eastern region of China including major urban and industrial areas"
    }
  ],
  "relationships": [
    {
      "source_entity": "Sentinel-5P",
      "target_entity": "TROPOMI",
      "relationship": "CARRIES",
      "relationship_description": "Sentinel-5P satellite platform carries the TROPOMI instrument"
    },
    {
      "source_entity": "TROPOMI",
      "target_entity": "NO2",
      "relationship": "MEASURES",
      "relationship_description": "TROPOMI sensor measures NO2 concentrations in the atmosphere"
    },
    {
      "source_entity": "NO2",
      "target_entity": "NOx",
      "relationship": "IS_TYPE_OF",
      "relationship_description": "NO2 is a type of nitrogen oxide (NOx)"
    },
    {
      "source_entity": "NO2",
      "target_entity": "Transportation",
      "relationship": "EMITTED_BY",
      "relationship_description": "NO2 is emitted by traffic and transportation sources"
    },
    {
      "source_entity": "Transportation",
      "target_entity": "Eastern China",
      "relationship": "LOCATED_IN",
      "relationship_description": "The analyzed traffic emissions are located in Eastern China"
    }
  ]
}

**Example Input 2:**
"We applied the DOAS algorithm in the UV spectral range to retrieve tropospheric ozone, which affects air quality and human health."

**Expected Output:**
{
  "entities": [
    {
      "entity_name": "DOAS",
      "entity_type": "RetrievalAlgorithm",
      "entity_description": "Differential Optical Absorption Spectroscopy, algorithm for retrieving trace gas concentrations from spectral measurements"
    },
    {
      "entity_name": "UV",
      "entity_type": "SpectralBand",
      "entity_description": "Ultraviolet electromagnetic spectrum band, wavelength range approximately 10-400 nm"
    },
    {
      "entity_name": "O3",
      "entity_type": "PolluingGas",
      "entity_description": "Tropospheric Ozone, secondary pollutant formed from photochemical reactions, harmful to air quality and human health"
    },
    {
      "entity_name": "Air Quality",
      "entity_type": "ApplicationDomain",
      "entity_description": "Research domain focusing on atmospheric pollution and its impacts on environmental quality"
    },
    {
      "entity_name": "Human Health",
      "entity_type": "ApplicationDomain",
      "entity_description": "Research domain focusing on health impacts of atmospheric pollutants"
    }
  ],
  "relationships": [
    {
      "source_entity": "DOAS",
      "target_entity": "UV",
      "relationship": "OPERATES_IN_BAND",
      "relationship_description": "DOAS algorithm operates in the UV spectral band for trace gas retrieval"
    },
    {
      "source_entity": "UV",
      "target_entity": "O3",
      "relationship": "SENSITIVE_TO",
      "relationship_description": "UV spectral band is sensitive to ozone absorption features"
    },
    {
      "source_entity": "O3",
      "target_entity": "Air Quality",
      "relationship": "AFFECTS_AIR_QUALITY",
      "relationship_description": "Tropospheric ozone is a criteria pollutant that negatively affects air quality"
    },
    {
      "source_entity": "O3",
      "target_entity": "Human Health",
      "relationship": "HEALTH_IMPACT",
      "relationship_description": "Ozone exposure causes respiratory and cardiovascular health effects"
    }
  ]
}

=====INPUT TEXT=====
{text}
=====OUTPUT=====
Return only valid JSON with "entities" and "relationships" keys. No additional text or explanations.
"""
