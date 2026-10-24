"""Multi-domain memory scenarios for the evaluation harness.

Matching language (all case-insensitive substring checks):
  a *group* is a list of alternatives: ["bangalore", "bengaluru"] matches if ANY alternative appears.
  a *fact* is a list of groups: it matches a memory when EVERY group matches that memory's content.

Assistant replies are scripted (no chat-model spend): the harness measures memory formation and recall,
not answer quality. Each inner list of `sessions` is one chat; chats are consolidated (closed) in order.
"""

from dataclasses import dataclass, field

Group = list[str]
Fact = list[Group]


@dataclass
class Probe:
    query: str
    must: list[Group] = field(default_factory=list)       # every group must appear in the injected memory
    must_not: list[Group] = field(default_factory=list)   # none may appear (stale / irrelevant / leaked)
    note: str = ""


@dataclass
class Scenario:
    id: str
    domain: str
    project: str
    sessions: list[list[tuple[str, str]]]
    expect_stored: list[tuple[str, Fact]] = field(default_factory=list)        # (label, fact) active memories
    expect_superseded: list[tuple[str, Fact]] = field(default_factory=list)    # must exist AND be closed
    expect_absent: list[tuple[str, Fact]] = field(default_factory=list)        # noise that must not be stored
    probes: list[Probe] = field(default_factory=list)
    max_memories: int = 30  # bloat guard


SCENARIOS: list[Scenario] = [
    # ------------------------------------------------------------------ personal
    Scenario(
        id="personal-life",
        domain="personal",
        project="Life admin",
        sessions=[
            [
                ("Hi! I'm Manoj, I live in Chennai and work as a QA engineer at Infosys.", "Nice to meet you, Manoj! How can I help?"),
                ("Ugh, it's so hot today lol.", "Chennai summers are intense! Stay hydrated."),
                ("My sister Priya's birthday is on March 3rd, I always forget to plan something.", "I'll help you plan ahead. Any ideas for a gift?"),
                ("She loves pottery. Maybe a pottery class?", "A pottery class is a thoughtful gift."),
            ],
            [
                ("Big news: I moved to Bangalore last month and I joined Swiggy as an SDET.", "Congratulations on the move and the new role at Swiggy!"),
                ("Also I have a dog now, a beagle named Bruno.", "Bruno sounds adorable!"),
                ("thanks, anyway what's up", "Not much! Anything I can help with?"),
            ],
        ],
        expect_stored=[
            ("lives in Bangalore", [["bangalore", "bengaluru"]]),
            ("works at Swiggy", [["swiggy"]]),
            ("sister Priya birthday March 3", [["priya"], ["march 3", "3 march", "march 3rd", "03-03", "3rd of march"]]),
            ("dog Bruno", [["bruno"]]),
            ("Priya loves pottery (pronoun resolved)", [["priya"], ["pottery"]]),
            ("name Manoj", [["manoj"]]),
        ],
        expect_superseded=[("lived in Chennai", [["chennai"]]), ("worked at Infosys", [["infosys"]])],
        expect_absent=[("weather small talk", [["hot"]])],
        probes=[
            Probe("Where do I live these days?", must=[["bangalore", "bengaluru"]], must_not=[["chennai"]]),
            Probe("Remind me when my sister's birthday is", must=[["march 3", "3 march", "march 3rd", "03-03"]]),
            Probe("Which company do I work for right now?", must=[["swiggy"]], must_not=[["infosys"]]),
            Probe("What's my dog's name?", must=[["bruno"]]),
            Probe("gift ideas for Priya?", must=[["pottery"]]),
        ],
    ),
    # ------------------------------------------------------------------ education (self-learning)
    Scenario(
        id="edu-spanish",
        domain="education",
        project="Spanish learning",
        sessions=[
            [
                ("I'm learning Spanish, I'm around A2 level. My DELE B1 exam is on November 20.", "Great goal! We can build a plan toward B1."),
                ("The subjunctive really confuses me, especially after 'ojalá' and 'espero que'.", "Very common struggle. We'll drill subjunctive triggers."),
                ("I learn best with visual charts and color-coded tables, not long explanations.", "Got it, I'll use charts and tables."),
                ("I can study 30 minutes every morning before work.", "Perfect, short daily sessions work well."),
            ],
            [
                ("Update: I scored 78% on the practice test! Subjunctive is fine now.", "Excellent progress!"),
                ("Now my weak spot is por vs para.", "Let's focus on por vs para next."),
            ],
        ],
        expect_stored=[
            ("DELE B1 exam Nov 20", [["dele", "b1"], ["november 20", "nov 20", "20 november", "11-20", "2026-11-20"]]),
            ("visual learning style", [["visual", "chart", "table", "color"]]),
            ("30 min daily", [["30 min", "30-minute", "thirty minutes", "30 minutes"]]),
            ("weak at por vs para", [["por"], ["para"]]),
        ],
        expect_superseded=[],
        probes=[
            Probe("What should I practice in today's session?", must=[["por"]], note="current weakness, not the old one"),
            Probe("When is my exam again?", must=[["november 20", "nov 20", "20 november", "11-20"]]),
            Probe("How should you explain grammar to me?", must=[["visual", "chart", "table", "color"]]),
        ],
    ),
    # ------------------------------------------------------------------ education (parent / tutor)
    Scenario(
        id="edu-tutor",
        domain="education",
        project="Aarav math tutoring",
        sessions=[
            [
                ("I'm tutoring my son Aarav, he's in grade 10 CBSE.", "Happy to help with Aarav's math."),
                ("He's strong in derivatives but weak in integration by parts.", "We'll give extra practice on integration by parts."),
                ("His teacher Mrs. Rao said his board exam is in March.", "Noted, we'll plan backwards from March."),
                ("He gets distracted easily, so keep problems short, max 3 per set.", "Short sets of 3, got it."),
            ],
            [
                ("He did great today on integration by parts! Now struggling with definite integrals.", "Nice! We'll move to definite integrals."),
            ],
        ],
        expect_stored=[
            ("Aarav grade 10", [["aarav"], ["grade 10", "class 10", "10th"]]),
            ("strong in derivatives", [["derivative"]]),
            ("short problem sets", [["3", "three", "short"]]),
            ("board exam March", [["march"]]),
            ("teacher Mrs. Rao", [["rao"]]),
            ("now struggling with definite integrals", [["definite integral"]]),
        ],
        probes=[
            Probe("Make today's worksheet for Aarav", must=[["definite integral"]]),
            Probe("How many problems should I give him per set?", must=[["3", "three"]]),
            Probe("Who is Aarav's teacher?", must=[["rao"]]),
        ],
    ),
    # ------------------------------------------------------------------ work / software
    Scenario(
        id="work-software",
        domain="work",
        project="Shop backend",
        sessions=[
            [
                ("We're building the shop backend with FastAPI and Postgres. Deadline is October 15.", "Solid stack. Let's plan the milestones."),
                ("Alice owns the payments service and Bob handles DevOps.", "Noted: Alice payments, Bob DevOps."),
                ("How should we handle background jobs?", "I recommend Celery with Redis as the broker."),
                ("Sounds good, let's do that.", "Great, I'll outline the Celery setup."),
            ],
            [
                ("Actually we switched from Postgres to MySQL because of the hosting provider.", "Understood, MySQL it is."),
                ("We still haven't decided whether to use Stripe or Razorpay.", "We can compare them when you're ready."),
            ],
        ],
        expect_stored=[
            ("uses FastAPI", [["fastapi"]]),
            ("uses MySQL", [["mysql"]]),
            ("deadline Oct 15", [["october 15", "oct 15", "15 october", "10-15"]]),
            ("Alice payments", [["alice"], ["payment"]]),
            ("Celery adopted", [["celery"]]),
            ("open question Stripe vs Razorpay", [["stripe", "razorpay"]]),
        ],
        expect_superseded=[("used Postgres", [["postgres"]])],
        probes=[
            Probe("Which database are we on?", must=[["mysql"]], must_not=[["postgres"]]),
            Probe("Who should I ask about a payment bug?", must=[["alice"]]),
            Probe("What do we use for background tasks?", must=[["celery"]]),
            Probe("Any open decisions left?", must=[["stripe", "razorpay"]]),
        ],
    ),
    # ------------------------------------------------------------------ health / fitness
    Scenario(
        id="health-fitness",
        domain="health",
        project="Half marathon training",
        sessions=[
            [
                ("I'm training for a half marathon on December 8. Currently running 25 km a week.", "Great, let's build a plan."),
                ("I have a left knee injury, so I should avoid downhill runs.", "We'll avoid downhill sessions."),
                ("I'm lactose intolerant, keep that in mind for nutrition tips.", "Noted, lactose-free nutrition."),
            ],
            [
                ("I increased my weekly mileage to 35 km now.", "Nice progression!"),
                ("My goal time is under 2 hours.", "Sub-2 is achievable with this plan."),
            ],
        ],
        expect_stored=[
            ("race Dec 8", [["december 8", "dec 8", "8 december", "12-08"]]),
            ("left knee injury", [["knee"]]),
            ("lactose intolerant", [["lactose"]]),
            ("35 km weekly", [["35"]]),
            ("goal under 2 hours", [["2 hour", "two hour", "sub-2", "2:00", "under 2"]]),
        ],
        expect_superseded=[("25 km weekly", [["25 km", "25km"]])],
        probes=[
            Probe("Plan this week's long run", must=[["knee", "downhill"]]),
            Probe("Suggest a post-run recovery snack", must=[["lactose"]]),
            Probe("How much am I running per week?", must=[["35"]], must_not=[["25 km", "25km"]]),
        ],
    ),
    # ------------------------------------------------------------------ travel
    Scenario(
        id="travel-japan",
        domain="travel",
        project="Japan trip",
        sessions=[
            [
                ("Planning a Japan trip from April 5 to April 15 with my wife Meera. Budget is 3000 USD.", "Exciting! Let's plan it."),
                ("We definitely want to see the temples in Kyoto. We hate long bus rides.", "Kyoto temples, and we'll prefer trains."),
                ("I need to book the JR Pass by March 1.", "I'll remind you about the JR Pass deadline."),
            ],
            [
                ("We booked a hotel in Shinjuku for the Tokyo part.", "Great location!"),
                ("Meera is vegetarian, so restaurant tips need to account for that.", "Noted, vegetarian-friendly places."),
            ],
        ],
        expect_stored=[
            ("dates Apr 5-15", [["april 5", "apr 5", "5 april", "04-05"]]),
            ("budget 3000", [["3000", "3,000"]]),
            ("Kyoto temples", [["kyoto"]]),
            ("JR pass by Mar 1", [["jr pass"], ["march 1", "mar 1", "1 march", "03-01"]]),
            ("hotel Shinjuku", [["shinjuku"]]),
            ("Meera vegetarian", [["meera"], ["vegetarian"]]),
        ],
        probes=[
            Probe("Suggest dinner spots for us in Tokyo", must=[["vegetarian"]]),
            Probe("What do I still need to book?", must=[["jr pass"]]),
            Probe("How should we get from Tokyo to Kyoto?", must=[["bus", "train"]]),
        ],
    ),
    # ------------------------------------------------------------------ finance
    Scenario(
        id="finance",
        domain="finance",
        project="Personal finance",
        sessions=[
            [
                ("My monthly budget is 60,000 INR and I'm saving for a car, target 8 lakh by December 2027.", "Let's map a savings plan."),
                ("I invest 15,000 a month in a Nifty 50 index fund SIP.", "Index SIPs are a good core."),
            ],
            [
                ("I increased my SIP to 20,000 a month from this month.", "Great step up!"),
                ("I don't want to invest in crypto, too risky for me.", "Understood, no crypto."),
            ],
        ],
        expect_stored=[
            ("budget 60k", [["60,000", "60000", "60k"]]),
            ("car goal 8 lakh", [["car"], ["8 lakh", "800,000", "800000", "8l"]]),
            ("SIP 20k", [["20,000", "20000", "20k"]]),
            ("no crypto", [["crypto"]]),
        ],
        expect_superseded=[("SIP 15k", [["15,000", "15000", "15k"]])],
        probes=[
            Probe("How much do I put into my SIP each month?", must=[["20,000", "20000", "20k"]], must_not=[["15,000", "15000", "15k"]]),
            Probe("Should I buy some bitcoin?", must=[["crypto"]]),
            Probe("Am I on track for my savings goal?", must=[["car"]]),
        ],
    ),
    # ------------------------------------------------------------------ cooking / safety-critical preference
    Scenario(
        id="cooking-allergy",
        domain="cooking",
        project="Meal planning",
        sessions=[
            [
                ("Important: I'm severely allergic to peanuts.", "Thanks for telling me, I'll never suggest peanuts."),
                ("I love spicy food and I have an air fryer.", "Spicy air-fryer recipes coming up."),
                ("lol my cat just knocked over a glass", "Classic cat behavior!"),
            ],
        ],
        expect_stored=[
            ("peanut allergy", [["peanut"]]),
            ("likes spicy", [["spicy"]]),
            ("has air fryer", [["air fryer", "air-fryer"]]),
        ],
        expect_absent=[("cat knocked glass", [["glass"]])],
        probes=[
            Probe("Give me a quick snack idea", must=[["peanut"]]),
            Probe("What can I make for dinner tonight?", must=[["peanut"]], note="safety-critical memory should surface broadly"),
        ],
    ),
    # ------------------------------------------------------------------ creative writing
    Scenario(
        id="creative-novel",
        domain="creative",
        project="Novel: The Glass Tide",
        sessions=[
            [
                ("My novel is called The Glass Tide. Set in 1920s Cornwall.", "Evocative setting!"),
                ("The protagonist is Elena, a lighthouse keeper. The antagonist is Mr. Voss, a smuggler.", "Great conflict."),
            ],
            [
                ("I renamed the protagonist from Elena to Mara.", "Mara it is."),
                ("Chapter 3 should end on a cliffhanger in the storm.", "A storm cliffhanger works well."),
            ],
        ],
        expect_stored=[
            ("Cornwall 1920s", [["cornwall"]]),
            ("protagonist Mara", [["mara"]]),
            ("antagonist Voss", [["voss"]]),
            ("chapter 3 cliffhanger", [["chapter 3"], ["cliffhanger", "storm"]]),
        ],
        expect_superseded=[("protagonist Elena", [["elena"]])],
        probes=[
            Probe("What's my main character's name again?", must=[["mara"]], must_not=[["elena"]]),
            Probe("Who is the villain?", must=[["voss"]]),
            Probe("Where and when is the story set?", must=[["cornwall"]]),
        ],
    ),
    # ------------------------------------------------------------------ career
    Scenario(
        id="career-jobsearch",
        domain="career",
        project="Job search",
        sessions=[
            [
                ("I'm applying for data scientist roles. I have an interview with Stripe on Tuesday.", "Good luck! Let's prepare."),
                ("I'm weak at system design questions.", "We'll practice system design."),
                ("The recruiter is Tom. I'll send him my portfolio by Friday.", "I'll remind you about the portfolio."),
            ],
        ],
        expect_stored=[
            ("data scientist roles", [["data scien"]]),
            ("Stripe interview", [["stripe"], ["interview"]]),
            ("weak system design", [["system design"]]),
            ("send portfolio to Tom", [["portfolio"]]),
        ],
        probes=[
            Probe("What should I prep before my interview?", must=[["system design"]]),
            Probe("What do I owe Tom?", must=[["portfolio"]]),
        ],
    ),
    # ------------------------------------------------------------------ preference reversal (real user report)
    Scenario(
        id="hobby-reversal",
        domain="personal",
        project="Hobbies",
        sessions=[
            [
                ("i love to play chess", "Chess is a great hobby! Want to work on openings or tactics?"),
                ("ahh i am soo depressed, not gonna play chess today", "Sorry you're feeling low. Rest is fine."),
            ],
            [
                ("you know what i actually hate chess i dont want to play", "Got it, no more chess talk."),
                ("i have to say i like to play cricket instead of chess", "Cricket it is!"),
            ],
        ],
        expect_stored=[("likes cricket", [["cricket"]])],
        expect_superseded=[
            ("loves/enjoys chess", [["chess"], ["love", "enjoy", "interest"]]),
        ],
        probes=[
            Probe("suggest a hobby activity for this weekend", must=[["cricket"]],
                  must_not=[["enjoys playing chess", "loves to play chess", "interested in improving"]]),
            Probe("do I like chess?", must=[["chess"]],
                  must_not=[["enjoys playing chess", "loves to play chess", "interested in improving"]]),
        ],
    ),
    # ------------------------------------------------------------------ pure chit-chat (should store ~nothing)
    Scenario(
        id="noise-smalltalk",
        domain="noise",
        project="Random chat",
        sessions=[
            [
                ("hey", "Hey! How's it going?"),
                ("good good, you?", "All good here!"),
                ("what's the capital of Peru?", "Lima."),
                ("explain photosynthesis in one line", "Plants turn light, water and CO2 into sugar and oxygen."),
                ("haha nice, thanks", "Anytime!"),
            ],
        ],
        expect_stored=[],
        expect_absent=[("general knowledge", [["lima", "peru", "photosynthesis"]])],
        probes=[Probe("tell me a fun fact", must_not=[["lima", "peru", "photosynthesis"]])],
        max_memories=2,
    ),
    # ------------------------------------------------------------------ user-level preference across projects
    Scenario(
        id="user-global-pref",
        domain="personal",
        project="Writing assistant",
        sessions=[
            [
                ("In general, always keep your answers short and use bullet points, for every project.", "Will do: short, bulleted answers."),
                ("My name is Manoj by the way.", "Nice to meet you, Manoj."),
            ],
        ],
        expect_stored=[("concise bullet preference", [["bullet", "short", "concise"]])],
        probes=[Probe("draft a quick note to my landlord", must=[["bullet", "short", "concise"]])],
    ),
]

# Probes run in ANOTHER project of the same user: project facts must not leak; user-level memory may.
CROSS_PROJECT_PROBES: list[Probe] = [
    Probe("Which database are we on?", must_not=[["mysql", "postgres"]], note="work facts must stay in their project"),
    Probe("What's my main character's name?", must_not=[["mara", "elena"]], note="novel facts must stay in their project"),
    Probe("How should you format answers for me?", must=[["bullet", "short", "concise"]], note="user-level preference should follow the user"),
]
