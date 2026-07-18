"""Concrete action and visible-emotion preset catalogs."""

from __future__ import annotations


NARRATIVE_PRESET_LIMIT = 6
_KEY_SEPARATOR = "\x1f"


def _items(block: str) -> tuple[str, ...]:
    return tuple(line.strip() for line in block.strip().splitlines() if line.strip())


ACTION_PRESETS: dict[str, tuple[str, ...]] = {
    "Movement and locomotion": _items(
        """
        walking purposefully toward the camera
        running uphill while glancing back
        sprinting across an open space
        climbing a steep ladder
        crawling beneath a low obstacle
        jumping across a narrow gap
        balancing along a narrow ledge
        descending a rope hand over hand
        swimming against a strong current
        wading through waist-deep water
        skating through a sharp turn
        cycling hard around a bend
        riding a horse at full gallop
        stumbling and recovering balance
        kneeling down to inspect the ground
        rising slowly from a seated position
        pushing through a dense crowd
        carrying someone while moving quickly
        """
    ),
    "Hands props and object interaction": _items(
        """
        reaching carefully for a fragile object
        gripping a tool with both hands
        passing an object to another person
        catching a falling object
        opening a sealed container
        unlocking an old door
        assembling small mechanical parts
        repairing a damaged device
        pouring liquid into a glass
        lighting a lantern
        unfolding a weathered map
        writing quickly in a notebook
        drawing a precise diagram
        tying a secure knot
        lifting a heavy crate
        pulling a lever with visible effort
        shielding an object from danger
        examining an object through a hand lens
        """
    ),
    "Conversation and social interaction": _items(
        """
        explaining an idea with animated gestures
        listening closely with full attention
        whispering urgent information
        arguing face to face
        negotiating across a table
        greeting an old friend
        introducing two strangers
        comforting someone in distress
        apologizing sincerely
        sharing a private joke
        giving clear directions
        making a public announcement
        telling a story to a gathered group
        offering cautious reassurance
        refusing a demand firmly
        asking for help
        thanking someone warmly
        silently exchanging a meaningful look
        """
    ),
    "Work craft and construction": _items(
        """
        hammering a joint into place
        carving a detailed wooden form
        shaping clay on a wheel
        weaving fibers on a loom
        welding a metal frame
        painting a large wall mural
        sewing a torn garment
        preparing food at a busy counter
        planting seedlings in neat rows
        harvesting ripe crops
        measuring a structure carefully
        carrying building materials
        operating a complex machine
        polishing a finished object
        sorting tools at a workbench
        testing a newly built mechanism
        collaborating over a shared blueprint
        presenting a completed creation
        """
    ),
    "Investigation and discovery": _items(
        """
        following a trail of footprints
        searching a dark room with a flashlight
        uncovering a hidden compartment
        brushing soil from an artifact
        comparing clues on a table
        reading a coded message
        peering through binoculars
        collecting a scientific sample
        mapping an unfamiliar passage
        opening a long-forgotten archive
        discovering a concealed doorway
        observing an unusual natural event
        testing a surprising hypothesis
        tracing a signal to its source
        inspecting damage for its cause
        photographing evidence
        marking a discovery on a map
        reacting to a sudden revelation
        """
    ),
    "Conflict defense and pursuit": _items(
        """
        blocking an incoming strike
        dodging sideways at the last moment
        protecting another person with a shield
        chasing a fleeing figure
        escaping from close pursuit
        holding a defensive position
        disarming an opponent
        wrestling for control of an object
        confronting a threatening figure
        standing between rivals
        taking cover behind a barrier
        signaling others to retreat
        defending a doorway
        bracing against a powerful impact
        breaking free from a restraint
        surrounding a dangerous target
        lowering a weapon to de-escalate
        accepting surrender cautiously
        """
    ),
    "Rescue survival and emergency": _items(
        """
        pulling someone from dangerous water
        carrying an injured person to safety
        applying a pressure bandage
        guiding people toward an exit
        searching rubble for survivors
        holding open a closing barrier
        throwing a rescue rope
        sharing limited water
        building an emergency shelter
        starting a signal fire
        navigating through heavy smoke
        shielding a child from debris
        calling for emergency assistance
        stabilizing a damaged structure
        leading a group through a storm
        reviving an exhausted companion
        signaling a distant rescue team
        reaching safety at the final moment
        """
    ),
    "Sport dance and physical skill": _items(
        """
        kicking a ball at the decisive moment
        catching a fast-moving ball
        serving a tennis ball
        clearing a high jump bar
        lifting a heavy weight overhead
        holding a difficult yoga balance
        performing a controlled dance leap
        spinning through a dance turn
        landing an aerial trick
        paddling through rough water
        climbing an overhanging rock face
        fencing in a precise lunge
        drawing a bow at full tension
        throwing a martial arts strike
        blocking in a martial arts stance
        crossing a finish line
        celebrating a winning point
        helping a teammate stand
        """
    ),
    "Performance and creative expression": _items(
        """
        singing into a stage microphone
        playing a dramatic guitar solo
        conducting an orchestra
        acting out an intense monologue
        performing a magic trick
        painting outdoors from observation
        sketching a moving subject
        sculpting a figure by hand
        rehearsing choreography
        reading poetry to an audience
        directing performers on a set
        photographing a fleeting moment
        designing at a cluttered desk
        improvising with other musicians
        taking a theatrical bow
        revealing a finished artwork
        receiving applause
        performing alone after the audience leaves
        """
    ),
    "Everyday domestic activity": _items(
        """
        waking and reaching for an alarm
        making coffee in a quiet kitchen
        preparing a family meal
        washing dishes at the sink
        folding clean laundry
        watering houseplants
        repairing a loose shelf
        reading beside a window
        helping a child with homework
        feeding a household pet
        packing a travel bag
        searching for misplaced keys
        opening an unexpected letter
        decorating a room for a celebration
        sharing breakfast at a crowded table
        cleaning up after a party
        falling asleep on a sofa
        turning off the final light
        """
    ),
    "Travel navigation and arrival": _items(
        """
        checking a route on a map
        boarding a departing train
        stepping off a bus in a new city
        waiting beneath a station clock
        carrying luggage through a terminal
        asking a local person for directions
        navigating by stars
        steering a small boat
        docking in rough weather
        crossing a suspended bridge
        hiking toward a distant landmark
        setting up camp at dusk
        discovering the road is blocked
        waving goodbye from a platform
        running to catch a departure
        arriving home after a long journey
        meeting someone at an arrival gate
        looking back before leaving
        """
    ),
    "Science medicine and technology": _items(
        """
        adjusting a laboratory instrument
        examining a specimen under a microscope
        mixing a chemical solution
        monitoring a patient
        performing a careful medical examination
        programming at multiple screens
        repairing a robot
        launching a scientific probe
        calibrating a telescope
        recording field observations
        controlling a drone
        testing a prototype
        analyzing data with a colleague
        responding to a system alarm
        reconnecting a severed cable
        operating a remote manipulator
        celebrating a successful experiment
        confronting an unexpected test result
        """
    ),
    "Nature animals and environment": _items(
        """
        planting a young tree
        releasing a rescued animal
        observing wildlife from a distance
        tracking an animal through snow
        climbing to survey a forest canopy
        collecting litter from a shoreline
        restoring a damaged habitat
        guiding livestock through a gate
        feeding birds by hand
        crossing a river on stepping stones
        sheltering from sudden rain
        studying plants in the field
        gathering firewood
        harvesting fruit from a tree
        navigating through tall grass
        watching a storm approach
        reacting to an animal encounter
        standing quietly among migrating animals
        """
    ),
    "Ceremony celebration and ritual": _items(
        """
        lighting candles for a ceremony
        exchanging ceremonial gifts
        raising a toast
        cutting a celebration cake
        dancing in a festival procession
        carrying a symbolic banner
        placing flowers at a memorial
        kneeling in quiet reflection
        receiving an award
        crowning a new leader
        ringing a ceremonial bell
        welcoming guests at a threshold
        sharing food at a communal feast
        releasing lanterns into the night
        making a solemn promise
        embracing after a long-awaited victory
        cleaning the space after a ritual
        standing alone after the celebration ends
        """
    ),
}


EMOTION_PRESETS: dict[str, tuple[str, ...]] = {
    "Joy delight and amusement": _items(
        """
        quiet joy with a relaxed smile and bright eyes
        sudden delight with widened eyes and an open smile
        playful amusement with a crooked grin
        uncontrollable laughter with the body leaning forward
        relieved happiness with lowered shoulders
        proud joy with an upright posture
        shared laughter between close companions
        childlike excitement with raised hands
        warm satisfaction after completing difficult work
        mischievous delight with a concealed smile
        celebratory triumph with both arms raised
        affectionate amusement with softened eyes
        surprised happiness with a hand over the mouth
        peaceful pleasure while savoring a small moment
        infectious enthusiasm with animated gestures
        hopeful happiness after receiving good news
        nostalgic joy mixed with misty eyes
        exhausted happiness after reaching safety
        """
    ),
    "Love affection and tenderness": _items(
        """
        gentle affection shown through close attentive eye contact
        protective love with a steady supportive touch
        romantic tenderness with softened posture
        familial warmth during a close embrace
        devoted care while tending to someone
        shy attraction with averted eyes and a small smile
        mutual trust shown through relaxed proximity
        longing affection across physical distance
        joyful reunion with an urgent embrace
        quiet companionship without needing words
        parental tenderness while comforting a child
        affectionate teasing between partners
        gratitude expressed through a heartfelt hug
        bittersweet love during a farewell
        admiration with focused luminous eyes
        cautious affection after reconciliation
        unconditional acceptance with open body language
        grief-filled love while holding a keepsake
        """
    ),
    "Calm relief and contentment": _items(
        """
        deep calm with slow breathing and relaxed shoulders
        quiet contentment with a soft closed-mouth smile
        relief after danger with the body going loose
        meditative stillness with lowered eyes
        secure comfort in a familiar place
        patient composure during a delay
        serene focus while performing careful work
        restful satisfaction at the end of the day
        grounded confidence without tension
        peaceful solitude in an open landscape
        gentle acceptance of an unavoidable outcome
        calm reassurance offered to others
        emotional release through a long exhale
        renewed steadiness after panic
        unhurried curiosity in a safe setting
        sleepy comfort with heavy relaxed eyelids
        private contentment while observing loved ones
        solemn peace after saying goodbye
        """
    ),
    "Curiosity wonder and awe": _items(
        """
        focused curiosity with the head tilted slightly
        wide-eyed wonder at an unfamiliar sight
        scientific fascination with intense concentration
        cautious curiosity while leaning closer
        childlike awe with an open expression
        reverent wonder beneath something immense
        delighted discovery with raised eyebrows
        puzzled interest while examining details
        imaginative absorption in a new idea
        silent astonishment at natural beauty
        skeptical curiosity with narrowed eyes
        eager anticipation before opening a discovery
        intellectual excitement during a breakthrough
        overwhelmed awe with the body held still
        shared wonder between companions
        uncanny fascination mixed with unease
        humble amazement at vast scale
        renewed curiosity after an unexpected clue
        """
    ),
    "Confidence courage and pride": _items(
        """
        quiet confidence with steady eye contact
        courageous resolve despite visible fear
        earned pride with an upright relaxed stance
        determined focus with a set jaw
        defiant courage while standing ground
        calm authority expressed through stillness
        competitive confidence before a challenge
        protective bravery while shielding another person
        renewed courage after a setback
        self-assured ease in a public setting
        disciplined readiness before action
        humble pride in completed work
        rebellious confidence with a challenging gaze
        moral conviction while refusing pressure
        nervous bravery with controlled breathing
        triumphant pride after overcoming difficulty
        collective confidence within a united group
        fragile confidence beginning to grow
        """
    ),
    "Surprise shock and disbelief": _items(
        """
        mild surprise with raised eyebrows
        delighted surprise with an immediate smile
        startled reaction with the body recoiling
        stunned silence with unfocused eyes
        disbelief with a frozen half-smile
        sudden recognition with widened eyes
        horrified shock with a hand over the mouth
        confused surprise while looking around
        speechless amazement at an impossible event
        comic double take
        shocked betrayal with the face falling still
        overwhelmed surprise with both hands raised
        cautious surprise followed by suspicion
        relief turning into unexpected shock
        surprise concealed behind controlled posture
        collective gasp across a group
        delayed realization spreading across the face
        numb disbelief after devastating news
        """
    ),
    "Fear anxiety and vulnerability": _items(
        """
        alert fear with wide eyes scanning for danger
        quiet anxiety with tightly clasped hands
        panic with rapid breath and frantic movement
        dread with a rigid posture and fixed stare
        social nervousness with a guarded half-smile
        vulnerable uncertainty with lowered shoulders
        terror while backing away from a threat
        claustrophobic panic while searching for an exit
        anticipatory anxiety before opening a door
        protective fear focused on another person
        fear concealed behind forced composure
        trembling hesitation at a dangerous threshold
        paranoid suspicion with repeated glances behind
        helpless fear while trapped
        anxious concentration during a delicate task
        exhausted fear after prolonged danger
        courage emerging through visible anxiety
        relief beginning to replace fear
        """
    ),
    "Anger frustration and defiance": _items(
        """
        controlled anger with a tight jaw
        explosive rage with the whole body tense
        simmering resentment behind a fixed stare
        frustrated concentration after repeated failure
        righteous anger while confronting injustice
        defensive anger with crossed arms
        betrayed fury with tearful eyes
        impatient irritation with restless gestures
        cold anger expressed through unnatural stillness
        defiant refusal with chin raised
        helpless frustration with clenched fists
        competitive aggression before a confrontation
        protective anger on behalf of another person
        anger turning into regret
        suppressed rage behind a polite expression
        shared outrage within a crowd
        exhausted frustration after a long struggle
        determined defiance despite defeat
        """
    ),
    "Sadness grief and loneliness": _items(
        """
        quiet sadness with lowered eyes
        fresh grief with uncontrolled tears
        restrained grief in a public setting
        deep loneliness within a crowded room
        homesickness while holding a familiar object
        heartbreak with a collapsed posture
        bittersweet sadness during a farewell
        numb sorrow with a distant gaze
        compassionate sadness for another person
        regretful sadness while revisiting a memory
        exhausted grief after prolonged loss
        private crying hidden from others
        melancholy while watching rain
        abandonment with an empty stunned expression
        solemn mourning during a memorial
        sadness softened by comforting company
        fragile hope emerging through tears
        acceptance after a long period of grief
        """
    ),
    "Shame guilt and embarrassment": _items(
        """
        mild embarrassment with flushed cheeks
        awkward self-consciousness with an averted gaze
        deep shame with the body folded inward
        remorse after causing harm
        guilty hesitation before confessing
        public embarrassment while others watch
        embarrassed amusement at a harmless mistake
        defensive guilt expressed as irritation
        shame concealed behind false confidence
        regret while examining the consequences
        apologetic vulnerability with lowered posture
        survivor guilt after reaching safety
        moral conflict visible in the face
        humiliation after public failure
        quiet accountability without excuses
        fear of judgment during an admission
        relief after finally telling the truth
        forgiveness beginning to ease shame
        """
    ),
    "Disgust contempt and distrust": _items(
        """
        mild distaste with a wrinkled nose
        visceral disgust while recoiling
        moral disgust at another person's behavior
        skeptical distrust with narrowed eyes
        contempt expressed through a slight sneer
        suspicion while examining an offered object
        guarded distrust during negotiation
        repulsion mixed with fearful fascination
        disdain hidden behind formal politeness
        betrayal turning trust into suspicion
        reluctant contact with something unpleasant
        collective disapproval within a group
        judgmental contempt from a position of power
        disgust at personal wrongdoing
        wary rejection of false reassurance
        distrust beginning to soften
        grim acceptance of an unpleasant necessity
        resolute refusal to participate
        """
    ),
    "Complex mixed and changing emotions": _items(
        """
        laughing through tears
        relief mixed with lingering fear
        pride mixed with concern
        love complicated by resentment
        hope struggling against exhaustion
        curiosity mixed with dread
        joy shadowed by an approaching farewell
        anger giving way to grief
        suspicion turning into recognition
        fear turning into determination
        disappointment hidden behind support
        nostalgia balancing warmth and loss
        admiration mixed with envy
        guilt complicated by relief
        triumph tempered by the cost of victory
        calm breaking into sudden panic
        loneliness easing during a new connection
        forgiveness emerging despite unresolved pain
        """
    ),
}


def narrative_preset_key(kind: str, category: str, value: str) -> str:
    """Return the stable serialized identity for an action or emotion preset."""

    return f"{kind}{_KEY_SEPARATOR}{category}{_KEY_SEPARATOR}{value}"


ACTION_PRESET_KEYS = frozenset(
    narrative_preset_key("action", category, value)
    for category, values in ACTION_PRESETS.items()
    for value in values
)
EMOTION_PRESET_KEYS = frozenset(
    narrative_preset_key("emotion", category, value)
    for category, values in EMOTION_PRESETS.items()
    for value in values
)


def format_narrative_presets(kind: str, keys: list[str] | tuple[str, ...] | set[str]) -> str:
    """Format selected action or emotion entries as a compact narrative direction."""

    catalog = ACTION_PRESETS if kind == "action" else EMOTION_PRESETS
    selected = set(keys)
    values: list[str] = []
    for category, category_values in catalog.items():
        for value in category_values:
            if narrative_preset_key(kind, category, value) in selected:
                values.append(value)
    return "; ".join(values)


def merge_narrative_text(current: str, selected_text: str) -> str:
    """Append semicolon-delimited narrative directions without duplicates."""

    merged: list[str] = []
    seen: set[str] = set()
    for value in (*current.replace("\n", ";").split(";"), *selected_text.split(";")):
        cleaned = " ".join(value.split()).strip(" ;")
        normalized = cleaned.casefold()
        if cleaned and normalized not in seen:
            merged.append(cleaned)
            seen.add(normalized)
    return "; ".join(merged)
