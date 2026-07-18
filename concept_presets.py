"""Broad content-concept catalog for prompt and comic ideation."""

from __future__ import annotations


CONCEPT_SELECTION_LIMIT = 8
_KEY_SEPARATOR = "\x1f"


def _items(block: str) -> tuple[str, ...]:
    return tuple(line.strip() for line in block.strip().splitlines() if line.strip())


CONCEPT_PRESETS: dict[str, tuple[str, ...]] = {
    "Character archetypes": _items(
        """
        reluctant hero
        seasoned mentor
        curious apprentice
        principled rebel
        charming trickster
        tragic antihero
        wandering guardian
        ambitious rival
        compassionate healer
        brilliant inventor
        eccentric scholar
        fearless explorer
        weary survivor
        idealistic reformer
        enigmatic stranger
        loyal companion
        fallen monarch
        reluctant leader
        misunderstood outsider
        determined underdog
        comic sidekick
        unreliable narrator
        """
    ),
    "Professions and social roles": _items(
        """
        astronaut
        archaeologist
        architect
        artisan
        botanist
        cartographer
        chef
        courier
        detective
        engineer
        farmer
        firefighter
        fisher
        journalist
        librarian
        mechanic
        musician
        paramedic
        photographer
        scientist
        street vendor
        teacher
        """
    ),
    "Relationships and groups": _items(
        """
        parent and child
        siblings
        lifelong friends
        new allies
        friendly rivals
        mentor and student
        leader and team
        found family
        estranged relatives
        reunited companions
        traveling party
        neighborhood community
        creative collective
        rescue team
        research expedition
        intergenerational household
        human and animal bond
        human and robot friendship
        diplomatic delegation
        secret society
        festival crowd
        solitary figure among strangers
        """
    ),
    "Actions and activities": _items(
        """
        climbing
        cooking
        dancing
        discovering
        drawing
        escaping
        exploring
        gardening
        gathering
        hiking
        inventing
        navigating
        performing
        repairing
        rescuing
        researching
        sailing
        storytelling
        teaching
        trading
        training
        traveling
        """
    ),
    "Human states and life moments": _items(
        """
        first day in a new place
        unexpected reunion
        quiet realization
        difficult farewell
        personal breakthrough
        shared celebration
        peaceful solitude
        creative flow
        culture shock
        homesickness
        courageous decision
        moral dilemma
        recovery after hardship
        adapting to change
        learning a new skill
        protecting a secret
        earning trust
        loss of certainty
        renewed hope
        rite of passage
        everyday resilience
        moment of wonder
        """
    ),
    "Animals and wildlife": _items(
        """
        arctic fox
        barn owl
        black bear
        blue whale
        butterfly swarm
        capybara
        coral reef fish
        crow
        deer
        elephant
        fireflies
        giant tortoise
        honeybees
        humpback whale
        jellyfish
        kingfisher
        octopus
        red panda
        snow leopard
        urban fox
        wild horses
        wolf pack
        """
    ),
    "Mythical beings and folklore": _items(
        """
        ancestral spirit
        benevolent dragon
        centaur
        changeling
        dryad
        elemental guardian
        fairy court
        forest giant
        gargoyle
        ghost ship crew
        griffin
        household spirit
        kelpie
        merfolk
        moon rabbit
        phoenix
        selkie
        shapeshifter
        stone golem
        thunderbird
        trickster spirit
        unicorn
        """
    ),
    "Science fiction entities and technology": _items(
        """
        android citizen
        biomechanical organism
        climate restoration machine
        companion robot
        cybernetic explorer
        deep-space probe
        digital consciousness
        drone swarm
        generation starship
        holographic archive
        interstellar habitat
        lunar research colony
        nanotechnology
        orbital elevator
        planetary rover
        quantum communication
        sentient spacecraft
        solar sail
        terraforming station
        underwater research habitat
        virtual city
        wearable exoskeleton
        """
    ),
    "Environments and biomes": _items(
        """
        alpine meadow
        arctic ice field
        bamboo forest
        bioluminescent cave
        cloud forest
        coral reef
        desert oasis
        freshwater wetland
        geothermal valley
        grassland savanna
        kelp forest
        mangrove coast
        misty moor
        monsoon jungle
        red rock canyon
        salt flat
        temperate rainforest
        tidal pool
        tropical island
        tundra
        volcanic landscape
        wildflower prairie
        """
    ),
    "Architecture and built spaces": _items(
        """
        ancient amphitheater
        art deco skyscraper
        brutalist civic center
        cliffside monastery
        covered market
        desert citadel
        floating village
        glass conservatory
        gothic cathedral
        industrial warehouse
        lighthouse
        mountain observatory
        old railway station
        organic architecture
        pedestrian megastructure
        rooftop settlement
        solar punk neighborhood
        stilt house community
        subterranean city
        vernacular courtyard house
        vertical garden tower
        waterside temple
        """
    ),
    "Interiors and functional spaces": _items(
        """
        artist studio
        botanical laboratory
        bustling kitchen
        clockmaker workshop
        cozy reading room
        emergency command center
        grand hotel lobby
        greenhouse interior
        home observatory
        hospital corridor
        industrial control room
        maker space
        museum archive
        night train compartment
        recording studio
        research library
        spacecraft cockpit
        subterranean workshop
        theater backstage
        traditional bathhouse
        vintage repair shop
        weather station
        """
    ),
    "Objects props and tools": _items(
        """
        antique compass
        astronomical chart
        battered suitcase
        ceremonial mask
        clockwork mechanism
        field journal
        folding map
        hand lens
        heirloom key
        musical instrument
        navigation sextant
        old camera
        portable radio
        protective helmet
        scientific specimen
        signal lantern
        sketchbook
        telescope
        tool belt
        travel trunk
        weathered book
        woven basket
        """
    ),
    "Vehicles and transportation": _items(
        """
        airship
        bicycle caravan
        cargo bicycle
        classic motorcycle
        deep-sea submersible
        desert crawler
        electric city bus
        expedition truck
        fishing boat
        funicular railway
        high-speed train
        horse-drawn wagon
        icebreaker ship
        lunar rover
        mountain cable car
        orbital shuttle
        river ferry
        sailboat
        solar-powered aircraft
        steam locomotive
        street tram
        vintage seaplane
        """
    ),
    "Plants fungi and natural forms": _items(
        """
        ancient olive tree
        baobab tree
        bioluminescent fungi
        bonsai
        carnivorous plants
        cherry blossoms
        coral formations
        giant ferns
        hanging moss
        kelp canopy
        lotus pond
        medicinal herb garden
        mushroom colony
        old-growth redwoods
        orchid collection
        overgrown ivy
        seed pods
        spiral seashells
        sunflower field
        tangled roots
        terraced rice plants
        wildflower meadow
        """
    ),
    "Narrative situations and conflicts": _items(
        """
        race against time
        rescue during a storm
        discovery of a hidden place
        defense of a community
        journey home
        search for a missing person
        survival in an unfamiliar world
        peaceful first contact
        contested inheritance
        forbidden expedition
        scientific mystery
        ecological restoration
        citywide blackout
        evacuation before disaster
        fragile truce
        mistaken identity
        secret revealed
        impossible delivery
        final performance
        rebuilding after catastrophe
        competition between inventors
        ordinary day disrupted
        """
    ),
    "Symbols and visual metaphors": _items(
        """
        bridge between worlds
        broken chain
        candle in darkness
        closed door
        cracked mirror
        empty chair
        endless staircase
        growing tree
        labyrinth
        lighthouse as guidance
        mask and identity
        melting clock
        open cage
        path splitting in two
        phoenix rebirth
        red thread of connection
        roots and belonging
        seed of possibility
        shadow self
        ship in a bottle
        threshold crossing
        window of opportunity
        """
    ),
    "Historical eras and worldbuilding": _items(
        """
        prehistoric hunter-gatherer life
        Bronze Age port city
        Iron Age hill settlement
        ancient river civilization
        classical Mediterranean city
        Silk Road caravan era
        early medieval village
        high medieval trade town
        Renaissance workshop
        age of sail
        Enlightenment scientific salon
        early industrial city
        Belle Epoque metropolis
        interwar modernism
        postwar reconstruction
        1960s space age
        1980s computer culture
        near-future circular economy
        far-future post-scarcity society
        alternate-history metropolis
        timeless mythic kingdom
        archaeological reconstruction
        """
    ),
    "Fashion costume and adornment": _items(
        """
        ceremonial robes
        climate-adaptive clothing
        embroidered folk costume
        expedition gear
        futuristic workwear
        historical court dress
        layered desert clothing
        minimalist tailoring
        protective rainwear
        recycled-material fashion
        retro sportswear
        ritual body paint
        sculptural haute couture
        seafaring work clothes
        spaceflight suit
        street fashion
        traditional weaving
        utilitarian jumpsuit
        vintage evening wear
        wearable technology
        winter survival clothing
        handcrafted jewelry
        """
    ),
    "Materials crafts and fabrication": _items(
        """
        basket weaving
        blown glass
        carved stone
        cast bronze
        ceramic glazing
        clockmaking
        embroidery
        forged iron
        handmade paper
        inlaid wood
        leatherworking
        mosaic
        natural dyeing
        origami
        porcelain
        recycled plastic
        stained glass
        textile printing
        timber joinery
        woven fiber
        bioplastic fabrication
        robotic manufacturing
        """
    ),
    "Food drink and culinary culture": _items(
        """
        artisan bakery
        ceremonial tea
        communal feast
        farmers market produce
        fermentation workshop
        food truck culture
        forest foraging
        heritage grain
        open-fire cooking
        orchard harvest
        pastry craft
        picnic
        rooftop garden meal
        seasonal cuisine
        spice market
        street food stall
        traditional noodle making
        tropical fruit harvest
        underwater restaurant
        vegetable garden
        winter soup kitchen
        zero-waste kitchen
        """
    ),
    "Product and graphic design": _items(
        """
        album cover
        book cover
        brand mascot
        collectible figurine
        editorial illustration
        educational poster
        exhibition display
        furniture concept
        icon system
        information dashboard
        magazine spread
        packaging design
        product hero image
        public wayfinding
        retro travel poster
        scientific infographic
        signage system
        sustainable appliance
        title sequence
        toy design
        transit map
        wearable product
        """
    ),
    "Abstract systems and spatial ideas": _items(
        """
        balance and imbalance
        cause and effect
        chaos becoming order
        collective memory
        connection across distance
        cyclical time
        emergence
        fragmentation and repair
        hidden network
        hierarchy and equality
        inside and outside
        memory palace
        nested worlds
        parallel realities
        pattern within randomness
        repetition and variation
        scale contrast
        self-similarity
        transformation
        visible sound
        worlds within worlds
        boundary dissolution
        """
    ),
}


def concept_preset_key(category: str, value: str) -> str:
    """Return the stable serialized identity for one catalog entry."""

    return f"{category}{_KEY_SEPARATOR}{value}"


CONCEPT_PRESET_KEYS = frozenset(
    concept_preset_key(category, value)
    for category, values in CONCEPT_PRESETS.items()
    for value in values
)


def format_concept_presets(keys: list[str] | tuple[str, ...] | set[str]) -> str:
    """Format selected keys as the comma-separated concept field contract."""

    selected = set(keys)
    values: list[str] = []
    seen: set[str] = set()
    for category, category_values in CONCEPT_PRESETS.items():
        for value in category_values:
            if concept_preset_key(category, value) not in selected:
                continue
            normalized = value.casefold()
            if normalized not in seen:
                values.append(value)
                seen.add(normalized)
    return ", ".join(values)


def merge_concept_text(current: str, selected_text: str) -> str:
    """Append selected concepts without duplicating existing comma-separated values."""

    merged: list[str] = []
    seen: set[str] = set()
    for value in (*current.split(","), *selected_text.split(",")):
        cleaned = " ".join(value.split())
        normalized = cleaned.casefold()
        if cleaned and normalized not in seen:
            merged.append(cleaned)
            seen.add(normalized)
    return ", ".join(merged)
