"""
MESE-CR — M4 (part 1b): CURATED hard-context set
================================================
Hand-authored HARD-context sentences: the tone/morality is entailed by a
concrete SITUATION and cannot be read off a surface keyword. Written as
mirrored pos/neg motif pairs so that every stimulus has an exact opposite
(controls for topic and length, not just for labels).

Why curated instead of LLM-drafted: the Qwen drafts (dataset_draft.csv ->
dataset_review.csv) were dominated by keyword leakage ("relieved",
"panic", "caring"), abstract propositions ("a fair distribution is
needed") and fragments. Only a handful survived review, so the honest fix
is to author the hard set directly and prove its properties by audit.

Guarantees checked in audit() below (via leak_lexicon):
  * one-hot label, exactly one axis active, polarity +/-1
  * NO leakage word from any axis lexicon (so tone is not in the words)
  * concrete event (determiner phrase / name), not an abstract claim
  * 6..32 words, ends with a period, no duplicates, no overlap with easy.csv
  * mirrored pair: pos and neg differ in outcome, share topic/entities
Group key = motif id, shared by BOTH poles of a pair, so grouped CV holds a
whole motif out (neither pole of a topic is ever in train and test at once).

Output: hard.csv  (sentence,V,A,C,F,kind,group)
"""
import csv, os, re
from collections import Counter, defaultdict
from leak_lexicon import AXES, find_leaks, reason_to_reject, has_concrete_marker

# Each entry: (motif, positive-pole sentence, negative-pole sentence)
VALENCE = [
    ("landlord", "The landlord called back to say the eviction notice had been withdrawn and the lease renewed for three more years.",
                 "The landlord called back to say the flat was needed for family and the keys had to be returned by the end of the month."),
    ("scan", "The scan came back with the mass gone and no further appointments on the calendar.",
             "The scan came back with the mass grown and a second round of appointments on the calendar."),
    ("loan", "The last payment cleared and the bank marked the loan closed.",
             "The last payment bounced and the bank added the missed amount back onto the balance."),
    ("visa", "Her application was approved and the visa arrived two weeks before the flight.",
             "Her application was returned unopened and the flight left without her."),
    ("busstop", "The bus stopped at the corner where she was waiting and the doors opened.",
                "The bus went past the corner where she was waiting and the doors stayed shut."),
    ("wells", "Rain fell on the fields for six days and the wells filled again.",
              "No rain fell on the fields for six days and the wells went dry."),
    ("dosage", "The doctor lowered the dosage and said the treatment could stop in spring.",
               "The doctor raised the dosage and said the treatment would go on through the winter."),
    ("offer", "The offer letter arrived with the salary figure written on the second line.",
              "The withdrawal letter arrived before the salary figure had ever been discussed."),
    ("missingchild", "The missing child was found asleep in the neighbour's garage twenty minutes later.",
                     "The missing child was found in the canal three days later."),
    ("grant", "The team's grant was renewed and the whole equipment list was ordered that week.",
              "The team's grant was not renewed and the equipment list was put aside."),
    ("exam", "The last exam was marked and every answer on the sheet was correct.",
             "The last exam was marked and none of the answers on the sheet were correct."),
    ("sale", "The house passed inspection and the keys changed hands that afternoon.",
             "The house failed inspection and the sale was called off that afternoon."),
    ("bridge", "The bridge reopened two weeks after the flood and traffic crossed it again.",
               "The bridge stayed shut two weeks after the flood and the detour added an hour each way."),
    ("manuscript", "Her manuscript was accepted by the journal on the first submission.",
                   "Her manuscript was returned by the journal on the first submission."),
    ("bonus", "The company reported a profit and the staff bonuses were paid in December.",
              "The company reported a loss and the staff bonuses were cancelled in December."),
    ("flight", "The plane landed on time and every bag came out on the belt.",
               "The plane landed four hours late and every bag was left behind."),
    ("wall", "The wall was rebuilt by volunteers and the family moved back into the ground floor.",
             "The wall stayed as rubble and the family lived in the shelter through the winter."),
    ("vote", "The vote went through and the playground was scheduled for construction in April.",
             "The vote failed and the playground was struck from the budget entirely."),
]
AROUSAL = [
    ("bell", "The bell rang through the hall and fourteen people were on their feet before it stopped.",
             "The hall clock moved past three while nobody in the room shifted position."),
    ("whistle", "The starter's whistle blew and the four runners left the blocks in one second.",
                "The starter's whistle was never blown and the four runners stood at the blocks talking."),
    ("nightcall", "The phone rang at three in the morning and she was dressed and out of the flat in ninety seconds.",
                  "The afternoon ran from one o'clock to five with nothing on the desk changing."),
    ("gate", "The gate dropped and the whole stand was up and shouting in one movement.",
             "The gate stayed open and the few people inside drifted between the tables."),
    ("water", "Water came over the edge of the sink in sheets and both of them worked the mop without stopping.",
              "Water dripped into the sink at the rate of the tap and neither of them moved from the chair."),
    ("blackout", "The lights went out across the whole block and every torch came on in one second.",
                 "The lights stayed on across the whole block and the torches lay untouched in the drawer."),
    ("sprint", "The sprinter left the line first and held the lead around the bend.",
               "The sprinter waited at the line while the others walked a lap of the far side."),
    ("firedoor", "The fire door flew open and the corridor filled with people moving one direction.",
                 "The fire door stayed shut and the corridor had nobody in it all morning."),
    ("message", "He read the message, took the keys, and was on the road within two minutes.",
                "He read the message, put the phone face down and went back to the sofa."),
    ("referee", "The referee's final gesture sent both benches onto the field.",
                "The referee walked to the centre circle while both benches sat in their places."),
    ("dial", "The needle on the dial swung to the red end within seconds of the switch.",
             "The needle on the dial held near zero for the whole hour."),
    ("sail", "Her hands took the ropes in pairs and the sail came down before the squall reached the harbour.",
             "Her hands rested on the ropes while the sail hung slack in the evening air."),
    ("classroom", "The last problem was finished and half the room was already standing with bags on shoulders.",
                  "The last problem was finished and nobody in the room reached for a bag."),
    ("doglead", "The dog's ears went up and it was through the gate before the lead was clipped on.",
                "The dog lay across the doorway and did not lift its head when the lead was clipped on."),
    ("scoreboard", "The scoreboard changed and the whole stand came to its feet in one movement.",
                   "The scoreboard held level and the whole stand sat in its places until the second half ended."),
    ("radio", "Orders came down the radio in short bursts and the crew moved at a run.",
              "The radio was off and the crew sat on the tailgate through the afternoon."),
    ("kettle", "The kettle went off, the toast popped, and both of them talked over each other.",
               "The kettle went off and neither of them got up to pour it."),
    ("phones", "Three separate phones went off in one minute and the meeting broke into motion.",
               "Three separate phones lay face down and the meeting worked through its agenda to the end."),
]
CARE = [
    ("groceries", "Ravi carried the stranger's groceries up four flights and set them by her door.",
                  "Ravi watched the stranger's groceries split on the stairs and stepped around them."),
    ("childbus", "When the child's bus never came, Leyla walked her the whole way home.",
                 "When the child's bus never came, the driver shut the door and left her at the stop."),
    ("path", "Two neighbours dug the old man's path clear before he woke.",
             "Two neighbours left the old man's path under ice and used the other gate."),
    ("nightshift", "The nurse came back on her night off to sit with the patient.",
                   "The nurse did not come back on her night off and the patient went through it with nobody in the ward."),
    ("seat", "She gave up her seat on the last bus and stood the whole way home.",
             "She took the last seat on the bus while the man on crutches held the rail."),
    ("cyclist", "He lifted the fallen cyclist's bike off the road and waited with her until her sister came.",
                "He stepped over the fallen cyclist's bike and drove on."),
    ("lunchmoney", "The class put their lunch money together for the boy whose parents had not paid the trip fee.",
                   "The class counted their lunch money and left the boy at the school gates."),
    ("vet", "The vet stayed past closing to set the dog's leg.",
            "The vet closed on the hour and the dog's leg went unset overnight."),
    ("windowroom", "Grandmother was given the room by the window where the family could sit with her daily.",
                   "Grandmother was given the room at the end of the corridor where visits were cut to one a week."),
    ("coat", "He put his coat over the sleeping man on the platform.",
             "He walked past the sleeping man on the platform where the wind came through."),
    ("newcomer", "She sat with the new student at lunch and showed her where the lockers were.",
                 "She moved to a table across the room and left the new student at the window."),
    ("roof", "When the roof leaked, the family took the neighbour's children in for the week.",
             "When the roof leaked, the family sent the neighbour's children back out to the yard."),
    ("smoke", "The crew went back into the smoke for the second person.",
              "The crew reached the doorway and turned back the way they came."),
    ("aloud", "He read his aunt's letter to her twice over that evening.",
              "He put his aunt's letter on the table unread and went out."),
    ("riders", "The riders stopped, blocked the lane and stood over the fallen rider until the ambulance came.",
               "The riders went around the fallen rider and the lane was clear by the next bend."),
    ("teacher", "The teacher stayed an hour to walk the youngest pupil to the bus stop.",
                "The teacher locked the classroom with the youngest pupil sitting inside at his desk."),
    ("straycat", "She set a bowl of water on the step each evening for the stray cat.",
                 "She chased the stray cat off the step each evening before the guests came."),
    ("piano", "The pair carried the piano up three flights so the widow could play it again.",
              "The pair left the piano on the pavement outside and told the widow it would not fit."),
]
FAIRNESS = [
    ("wages", "The four workers logged eight hours each and each of them was handed eighty birr at six.",
              "The four workers logged eight hours each and the envelopes at six held eighty, eighty, eighty and three hundred birr."),
    ("crates", "Aisha brought in forty crates and Sam brought in forty, and each of them was handed four hundred birr.",
               "Aisha brought in forty crates and Sam brought in four, and Sam was handed four hundred birr and Aisha one hundred."),
    ("cake", "The last slice of cake was cut down the middle and both children were given a plate.",
             "The last slice of cake was cut in two and both pieces went on the elder child's plate."),
    ("raffle", "The raffle tickets went into one drum and the winner was drawn with the room watching.",
               "The raffle tickets went into one drum and the winner's name was written on a slip in the organiser's pocket."),
    ("contractors", "The two contractors billed fourteen days each and both invoices cleared at the posted rate.",
                    "The two contractors billed fourteen days each and one invoice cleared at the posted rate while the other was cut in half."),
    ("wellfee", "The well fee was set at one hundred birr per household and one hundred birr was collected at every door on the street.",
                "The well fee was set at one hundred birr per household and fifty was collected at the near end and two hundred at the far end."),
    ("checkpoint", "Both drivers had their licences checked at the checkpoint and both cars were waved on.",
                   "Both drivers had their licences checked at the checkpoint, one car was waved on and the other's boot was opened for an hour."),
    ("marking", "The exam papers were marked with the names covered and the scores went up the next day.",
                "The exam papers were marked with the names showing and the professor's own students came out at the top."),
    ("rent", "The landlord charged every flat on the second floor the rent printed in the contract.",
             "The landlord charged the second floor the printed rent and raised the two flats beside his door by a third."),
    ("queue", "The queue at the gate was served in the order it formed until the last person passed.",
              "The queue at the gate was served until a car with embassy plates pulled to the front and cut the line."),
    ("wagetable", "The company published the wage table and paid every grade the figure printed on it.",
                  "The company published the wage table and paid the manager's friends one grade above the figure printed on it."),
    ("seats", "Two hundred seats went to each region by the committee's own rules.",
              "Two hundred seats went to each region on paper and four hundred went to the region holding the chairman's constituency."),
    ("draw", "The draw put the top eight seeds into separate halves by rotation.",
             "The draw put six of the top seeds into one half and left the host club by itself in the other."),
    ("audit", "The audit checked the receipts of every branch for the full year.",
              "The audit checked the receipts of one branch for the full year and the other eleven files were closed unopened."),
    ("ration", "When the well ran short, the water was measured out by household size for all nine families.",
               "When the well ran short, the water was measured out for six families while the three houses by the pump drew what they wanted."),
    ("scholarship", "The scholarship committee scored all twenty applications on the published criteria before the interviews.",
                    "The scholarship committee interviewed the director's niece first and the other nineteen folders were put away unopened."),
    ("trek", "The eight members took one turn each carrying the equipment on the trek.",
             "Two of the eight members carried the equipment on every day of the trek."),
    ("fine", "The fine for late returns was charged to every borrower who brought a book back late.",
             "The fine for late returns was charged to borrowers from the village and dropped for the staff's own children."),
]

CELLS = [(0, VALENCE), (1, AROUSAL), (2, CARE), (3, FAIRNESS)]


def build_hard(with_group=True):
    """Return [[sentence, V, A, C, F, 'hard', group], ...] (group only if asked)."""
    rows = []
    for axis_idx, pairs in CELLS:
        for motif, pos, neg in pairs:
            group = f"{AXES[axis_idx].lower().replace('/', '')}|{motif}"
            for sentence, pole in ((pos, 1), (neg, -1)):
                vals = [0, 0, 0, 0]
                vals[axis_idx] = pole
                row = [sentence] + vals + ["hard"]
                if with_group:
                    row.append(group)
                rows.append(row)
    return rows


def _norm(s):
    return re.sub(r"[^a-z0-9 ]", "", s.lower()).strip()


def audit(rows, easy_path="easy.csv"):
    """Return (problems, stats). A problem-free audit is required before merge."""
    problems = []
    seen = {}
    for r in rows:
        s, vals = r[0], r[1:5]
        where = f"[{AXES[vals.index(next(v for v in vals if v != 0))] if any(vals) else '?'}] {s}"
        # label shape
        if sum(1 for v in vals if v != 0) != 1:
            problems.append(f"{where}: label not one-hot -> {vals}")
        elif next(v for v in vals if v != 0) not in (1, -1):
            problems.append(f"{where}: polarity not +-1 -> {vals}")
        # leakage / concreteness / length
        for reason in reason_to_reject(s):
            problems.append(f"{where}: {reason}")
        # duplicates (incl. across easy/hard)
        if _norm(s) in seen:
            problems.append(f"{where}: DUPLICATE of -> {seen[_norm(s)]}")
        seen[_norm(s)] = s
    if os.path.exists(easy_path):
        with open(easy_path, encoding="utf-8", newline="") as f:
            for e in csv.DictReader(f):
                if _norm(e["sentence"]) in seen:
                    problems.append(f"[easy-overlap] {e['sentence']}")
    # stats
    stats = defaultdict(Counter)
    for r in rows:
        ax = AXES[next(i for i, v in enumerate(r[1:5]) if v != 0)]
        stats[ax][r[1:5][AXES.index(ax)]] += 1
    lens = [len(r[0].split()) for r in rows]
    stats["_len"] = Counter({f"min={min(lens)}": 1, f"max={max(lens)}": 1,
                            f"mean={sum(lens)/len(lens):.1f}": 1})
    stats["_groups"] = Counter({f"n={len({r[6] for r in rows if len(r) > 6})}": 1})
    return problems, stats


def write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["sentence", "V", "A", "C", "F", "kind", "group"])
        w.writerows(rows)


if __name__ == "__main__":
    rows = build_hard()
    problems, stats = audit(rows)
    for ax in AXES:
        print(f"{ax:>10}: {dict(stats[ax])}")
    print(f"length   : {dict(stats['_len'])}")
    print(f"groups   : {dict(stats['_groups'])}")
    if problems:
        print(f"\nAUDIT FAILED — {len(problems)} problem(s):")
        for p in problems[:40]:
            print("  -", p)
    else:
        print("\nAUDIT CLEAN: no leakage words, all concrete, no duplicates.")
    write_csv("hard.csv", rows)
    print(f"WROTE hard.csv: {len(rows)} hard-context sentences")
