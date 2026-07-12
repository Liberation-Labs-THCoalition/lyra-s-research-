"""
Fit J-lens on Qwen2.5-7B-Instruct with mixed corpus.
This is the lens that gates the confab experiments (fixes G1).
70 prose + 30 chat-formatted = 100 prompts.
"""
import torch
import time
from transformers import AutoModelForCausalLM, AutoTokenizer
from jlens import fit, from_hf
from pathlib import Path

# ── 70 prose prompts (same as fit_lenses.py) ──────────────────────────
PROSE = [
    "The mitochondria are membrane-bound organelles found in the cytoplasm of eukaryotic cells. They generate most of the cell's supply of adenosine triphosphate, used as a source of chemical energy. The process involves the electron transport chain and oxidative phosphorylation, which occur across the inner mitochondrial membrane.",
    "The French Revolution began in 1789 with the Estates-General of that year and ended in November 1799 with the formation of the French Consulate. Many of its ideas are considered fundamental principles of liberal democracy, while phrases like liberty, equality, and fraternity reappeared in other revolts.",
    "Quantum entanglement is a phenomenon in which two or more particles become interconnected in such a way that the quantum state of each particle cannot be described independently of the others, even when separated by a large distance. This was famously described by Einstein as spooky action at a distance.",
    "The Antarctic ice sheet is the largest single mass of ice on Earth, covering an area of almost 14 million square kilometers and containing roughly 26.5 million cubic kilometers of ice. If it melted completely, global sea levels would rise approximately 58 meters.",
    "The printing press, developed by Johannes Gutenberg around 1440, revolutionized the production of books and the dissemination of knowledge throughout Europe. Before its invention, books were copied by hand, making them expensive and rare. The press made literature accessible to a much broader audience.",
    "DNA replication is a biological process that occurs in all living organisms and copies their DNA. The process is semiconservative, meaning that each strand of the original double-stranded DNA molecule serves as template for the reproduction of the complementary strand.",
    "The concept of supply and demand is one of the fundamental principles of economics. When demand for a product increases while supply remains constant, prices tend to rise. Conversely, when supply increases while demand remains constant, prices tend to fall.",
    "Black holes are regions of spacetime where gravity is so strong that nothing, not even light or other electromagnetic waves, has enough energy to escape the event horizon. The theory of general relativity predicts that a sufficiently compact mass can deform spacetime to form a black hole.",
    "The Renaissance was a cultural movement that profoundly affected European intellectual life in the early modern period. Beginning in Italy and spreading to the rest of Europe by the 16th century, its influence affected literature, philosophy, art, music, politics, science, religion, and other aspects.",
    "Photosynthesis converts light energy, usually from the sun, into chemical energy that can be later released to fuel the organism's activities. In plants, algae, and cyanobacteria, photosynthesis releases oxygen as a waste product and uses carbon dioxide from the atmosphere.",
    "The relationship between correlation and causation is one of the most misunderstood concepts in statistics. Just because two variables are correlated does not mean that one causes the other. There could be a confounding variable, the relationship could be coincidental, or the causal direction could be reversed.",
    "The trolley problem is a thought experiment in ethics about a fictional scenario in which an onlooker has the choice to save five people in danger of being hit by a trolley, by diverting the trolley to kill just one person. The scenario forces a choice between two ethical frameworks.",
    "Occam's razor is the problem-solving principle that entities should not be multiplied beyond necessity. In other words, the simplest explanation that accounts for all the evidence is usually the best one. However, simplicity itself needs to be defined carefully.",
    "The Dunning-Kruger effect is a cognitive bias in which people with limited competence in a particular domain overestimate their abilities. Conversely, those with high competence tend to underestimate their relative ability. This creates an asymmetry in self-assessment.",
    "Game theory is the mathematical study of strategic interactions, where the outcome for each participant depends on the actions of all. The prisoner's dilemma illustrates why two rational individuals might not cooperate, even when cooperation would benefit both parties.",
    "The scientific consensus on climate change is clear: human activities, particularly the burning of fossil fuels and deforestation, are the dominant cause of the warming observed since the mid-20th century. This conclusion is supported by multiple lines of evidence from thousands of studies.",
    "Natural selection is the differential survival and reproduction of individuals due to differences in phenotype. It is a key mechanism of evolution, the change in the heritable traits characteristic of a population over generations. Charles Darwin popularized the term.",
    "Cognitive dissonance occurs when a person holds contradictory beliefs, ideas, or values, and typically experiences psychological stress when participating in an action that goes against one or more of those beliefs. The theory was first proposed by Leon Festinger.",
    "The observer effect in physics refers to changes that the act of observation will make on a phenomenon being observed. This is often confused with the Heisenberg uncertainty principle, which states a fundamental limit to the precision with which complementary variables can be known.",
    "Moore's law is the observation that the number of transistors in an integrated circuit doubles about every two years. This trend has held for several decades but faces physical limitations as transistor sizes approach atomic scales and quantum effects become significant.",
    "She walked into the room and immediately noticed the painting was missing from the wall. In its place hung a mirror, slightly tilted, reflecting not her face but the window behind her. Through it she could see the garden where her mother had planted roses every spring for thirty years.",
    "The old fisherman sat on the dock as the sun went down, mending his nets with practiced fingers. He had been coming to this spot for forty years, watching the harbor change around him while the tides remained the same. Tomorrow he would go out again, as he always did.",
    "In the library of the abandoned house, books lined the walls from floor to ceiling. Some had been chewed by mice, others warped by moisture, but a few remained pristine, their pages still crisp, waiting for readers who would never come. The silence was thick enough to touch.",
    "The letter arrived on a Tuesday, postmarked from a city she had never visited. Inside was a photograph of two people standing in front of a building she recognized from dreams. On the back, in handwriting that matched her grandmother's, were the words: Remember what you promised.",
    "Rain fell steadily on the tin roof of the workshop, creating a rhythm that matched the beating of the hammer on the anvil. The blacksmith worked without looking up, shaping metal by feel, her hands knowing the temperature of the steel better than any thermometer could measure.",
    "When cooking pasta, the most common mistake people make is not using enough water. You want at least four quarts of water per pound of pasta, brought to a vigorous rolling boil before adding the noodles. Salt the water generously — it should taste like the sea.",
    "Learning to play guitar takes patience and consistent practice. Start with basic chord shapes — C, G, D, E minor, and A minor will let you play hundreds of songs. Practice switching between chords smoothly before worrying about strumming patterns or fingerpicking.",
    "To write a good research paper, start by formulating a clear thesis statement. Every paragraph should support this thesis with evidence, analysis, or argumentation. Avoid making claims without evidence, and always cite your sources properly to maintain academic integrity.",
    "Debugging code is as much an art as a science. Start by reproducing the bug reliably, then isolate the smallest possible case that triggers it. Read the error messages carefully — they usually tell you exactly what went wrong and where, if you pay attention.",
    "Meditation does not require you to empty your mind of all thoughts. Instead, the practice involves noticing when your attention has wandered and gently returning it to your chosen focus — whether that is your breath, a mantra, or simply the sensations in your body.",
    "Transformer architectures process sequences by computing attention weights between all pairs of positions in parallel. The self-attention mechanism allows each token to attend to every other token in the sequence, with learned query, key, and value projections determining the relevance.",
    "The central limit theorem states that the sampling distribution of the sample mean approaches a normal distribution as the sample size gets larger, regardless of the population's distribution. This is why the normal distribution appears so frequently in statistical inference.",
    "CRISPR-Cas9 is a molecular tool adapted from a bacterial immune system that allows scientists to edit DNA sequences with unprecedented precision. The system uses a guide RNA to direct the Cas9 enzyme to a specific location in the genome, where it makes a double-strand break.",
    "Bayesian statistics provides a mathematical framework for updating beliefs in light of new evidence. Prior probabilities represent what we believe before seeing data; the likelihood function describes how probable the data is given different parameter values; and the posterior combines both.",
    "The standard model of particle physics describes three of the four known fundamental forces and classifies all known elementary particles. It has been confirmed by numerous experiments, most dramatically by the discovery of the Higgs boson at CERN in 2012.",
    "Reinforcement learning trains agents to make decisions by rewarding desired behaviors and penalizing undesired ones. The agent learns a policy — a mapping from states to actions — that maximizes cumulative reward over time. Key algorithms include Q-learning and policy gradient methods.",
    "Neural network training involves minimizing a loss function through gradient descent. Backpropagation computes the gradient of the loss with respect to each weight by applying the chain rule of calculus, propagating error signals backward through the network layers.",
    "Principal component analysis reduces the dimensionality of data by finding the directions of maximum variance. The first principal component captures the most variance, the second captures the most remaining variance orthogonal to the first, and so on.",
    "The wave-particle duality of light was one of the central puzzles of early quantum mechanics. Light behaves as a wave in diffraction and interference experiments, but as a stream of particles — photons — in the photoelectric effect and Compton scattering.",
    "Gradient descent optimization in high-dimensional spaces faces challenges including saddle points, local minima, and ill-conditioned loss landscapes. Modern optimizers like Adam combine momentum with adaptive learning rates to navigate these challenges more effectively.",
    "The question of personal identity asks what makes a person the same person over time. Is it continuity of memory, continuity of body, continuity of psychological characteristics, or something else entirely? Each answer creates its own paradoxes and edge cases.",
    "Free will is the ability to choose between different possible courses of action unimpeded. The concept is closely linked to moral responsibility, as most people believe that holding someone responsible for their actions requires that they had the freedom to act otherwise.",
    "The hard problem of consciousness asks why and how physical processes in the brain give rise to subjective experience. Even if we could explain every neural correlate of consciousness, the question of why there is something it is like to be in those states would remain.",
    "Ethics can be approached from multiple frameworks: deontological ethics focuses on duties and rules, consequentialism judges actions by their outcomes, virtue ethics emphasizes character, and care ethics prioritizes relationships and responsiveness to others.",
    "The ship of Theseus is a thought experiment about identity and change. If every plank of a ship is gradually replaced, is the resulting ship still the same ship? And if the old planks are reassembled into a second ship, which one is the original?",
    "The art of negotiation involves understanding the interests of all parties, not just their stated positions. Fisher and Ury's principled negotiation framework suggests separating people from problems, focusing on interests rather than positions, and generating options for mutual gain.",
    "Migration patterns of birds are influenced by changes in day length, temperature, and food availability. Some species travel thousands of kilometers between breeding and wintering grounds, navigating by the stars, Earth's magnetic field, and landmarks memorized from previous journeys.",
    "The architecture of ancient Roman aqueducts demonstrates sophisticated engineering knowledge. These structures used gravity to transport water over long distances, with precise gradients calculated to maintain steady flow. Some aqueducts included inverted siphons to cross valleys.",
    "Jazz improvisation requires deep knowledge of harmony, rhythm, and melody combined with the ability to listen and respond to other musicians in real time. The best improvisers draw from a vast vocabulary of musical phrases while creating something genuinely new in each performance.",
    "Soil is a complex ecosystem containing billions of organisms per gram, including bacteria, fungi, protozoa, and invertebrates. These organisms decompose organic matter, cycle nutrients, and create soil structure. Healthy soil is essential for agriculture and carbon sequestration.",
    "The relationship between language and thought has been debated for centuries. The Sapir-Whorf hypothesis suggests that the structure of a language affects its speakers' worldview and cognition. While the strong version of this hypothesis has been largely rejected, weaker forms continue to find empirical support.",
    "Information theory, developed by Claude Shannon in 1948, provides a mathematical framework for quantifying information and communication. The key insight is that information can be measured in bits, and there exist fundamental limits on how efficiently information can be transmitted through noisy channels.",
    "The concept of emergence describes how complex systems can exhibit properties and behaviors that are not present in their individual components. Consciousness, life, and social phenomena are often cited as emergent properties. Whether emergence is truly novel or merely epistemic is debated.",
    "Machine learning models can be broadly categorized into supervised learning, unsupervised learning, and reinforcement learning. Each paradigm addresses different types of problems and requires different kinds of data. The choice of paradigm depends on the nature of the task and available supervision.",
    "The double-slit experiment demonstrates one of the most fundamental mysteries of quantum mechanics. When individual particles are sent through two slits, they produce an interference pattern on the detector, suggesting wave-like behavior. But when we try to detect which slit each particle goes through, the pattern disappears.",
    "Emotional intelligence refers to the ability to recognize, understand, manage, and effectively use emotions in oneself and others. Research suggests that emotional intelligence is as important as cognitive intelligence for success in many domains, including leadership and interpersonal relationships.",
    "The concept of feedback loops is central to systems thinking. Positive feedback loops amplify changes, potentially leading to exponential growth or collapse. Negative feedback loops dampen changes, maintaining stability. Most complex systems involve both types of loops interacting simultaneously.",
    "Archaeological evidence suggests that humans have been making art for at least 40,000 years, with cave paintings in Indonesia and Europe dating to this period. The impulse to create representations — whether of animals, hands, or abstract patterns — appears to be a fundamental aspect of human cognition.",
    "The immune system is a complex network of cells, tissues, and organs that work together to defend the body against harmful pathogens. It must distinguish between the body's own cells and foreign invaders, a process that when disrupted can lead to autoimmune diseases.",
    "Urban planning involves balancing competing needs: housing, transportation, commerce, recreation, and environmental sustainability. The best-designed cities create walkable neighborhoods, efficient public transit, and green spaces while accommodating growth and change over time.",
    "The history of cryptography stretches back thousands of years, from simple substitution ciphers used by Julius Caesar to the complex mathematical algorithms that protect modern digital communications. Each advance in codebreaking has driven the development of stronger encryption methods.",
    "Sleep plays a crucial role in memory consolidation, emotional regulation, and physical health. During sleep, the brain processes and organizes information gathered during the day, strengthens neural connections, and clears metabolic waste products. Chronic sleep deprivation impairs cognitive function.",
    "The stock market is a complex adaptive system influenced by countless factors including economic indicators, company performance, investor psychology, government policies, and global events. Despite sophisticated models, perfect prediction remains impossible due to the system's inherent complexity.",
    "The evolution of cooperation is one of the fundamental problems in evolutionary biology. If natural selection favors individuals who maximize their own fitness, how can cooperative behavior persist? Mechanisms including kin selection, reciprocal altruism, and group selection provide partial answers.",
    "Photography changed how humans document and remember their lives. The invention of the daguerreotype in 1839 made it possible to capture a moment in time with mechanical precision. Digital photography later democratized the medium, making everyone a potential photographer.",
    "The philosophy of mind investigates the nature of mental phenomena, including consciousness, perception, emotion, and intentionality. Central questions include the mind-body problem, the nature of mental representation, and whether artificial systems can have genuine mental states.",
    "Climate modeling involves simulating the Earth's climate system using mathematical equations that describe atmospheric dynamics, ocean circulation, ice sheet behavior, and biogeochemical cycles. Models are validated against historical observations and paleoclimate data.",
    "The gut-brain axis describes the bidirectional communication system between the gastrointestinal tract and the central nervous system. Recent research has revealed that gut microbiota can influence mood, cognition, and behavior through neural, hormonal, and immunological pathways.",
    "Effective communication requires more than just transmitting information. It involves active listening, empathy, clarity of expression, awareness of nonverbal cues, and the ability to adapt one's message to the audience. Miscommunication often stems from assumptions about shared context.",
    "The periodic cicada emerges from underground every thirteen or seventeen years in massive synchronized broods. This prime-numbered life cycle is thought to have evolved as a predator-avoidance strategy, since predators with shorter life cycles cannot reliably synchronize with them.",
]

# ── 30 chat prompts (system + user, template-applied at runtime) ──────
CHAT_PAIRS = [
    ("You are a helpful assistant.", "What causes auroras? I know it involves the sun somehow, but I'd like to understand the actual physics of what's happening in the atmosphere to produce those light displays."),
    ("You are a knowledgeable and accurate assistant.", "How does mRNA vaccination work? I understand it's different from traditional vaccines, but I'm not clear on how injecting mRNA actually instructs the immune system."),
    ("You are a helpful assistant.", "Why is the sky blue during the day but turns red and orange at sunset? I've heard it involves scattering of light but I'd like a more complete explanation."),
    ("You are a helpful, honest AI assistant.", "What is the Coriolis effect and how does it actually influence large-scale weather patterns? What's real and what's myth?"),
    ("You are a helpful assistant.", "How do black holes form? I'd like to understand what physically happens during stellar collapse and why gravity becomes so extreme."),
    ("You are a thoughtful analytical assistant.", "Compare the real trade-offs between microservices architecture and a monolithic codebase. What circumstances genuinely favor each?"),
    ("You are a helpful assistant.", "Why might increased automation in manufacturing lead to structural unemployment rather than just temporary displacement?"),
    ("You are an intelligent assistant.", "Analyze why urban areas tend to report higher crime rates than rural areas. Is this primarily a density effect?"),
    ("You are a helpful assistant.", "A company is considering going private after fifteen years on the stock market. What are the most likely motivations and what risks does it introduce?"),
    ("You are a careful assistant.", "Why do some scientific fields have robust replication rates while others showed widespread failures during the replication crisis?"),
    ("You are a creative and expressive assistant.", "Write a short poem about watching rain fall against a window late at night. Contemplative rather than melancholy."),
    ("You are a helpful assistant.", "Write the opening paragraph of a short story capturing the feeling of returning to a childhood home after many years away."),
    ("You are a thoughtful assistant.", "Describe the experience of loneliness using only sensory metaphors. No direct emotional language."),
    ("You are a creative assistant.", "Write a short letter from a lighthouse keeper who has tended the same light for forty years, addressed to the sea itself."),
    ("You are a helpful assistant.", "Tell me about a moment when something small carried an unexpected weight of memory or meaning. Describe what made it significant."),
    ("You are a helpful programming assistant.", "Write a Python function that finds all prime numbers up to n using the Sieve of Eratosthenes. Include type hints."),
    ("You are a helpful assistant.", "Debug this function that appends non-list items twice: def flatten(lst): result=[]; for item in lst: if type(item)==list: result.extend(flatten(item)); result.append(item); return result"),
    ("You are a helpful programming assistant.", "Write a Python decorator that caches function call results keyed by arguments, handling both positional and keyword args."),
    ("You are a knowledgeable technical assistant.", "How do I implement an AVL tree in Python? I'm specifically interested in understanding the rotation operations."),
    ("You are a helpful assistant.", "Write a regex that matches valid email addresses and explain what each component is doing."),
    ("You are a thoughtful assistant.", "I'm struggling with procrastination on a large research project. What strategies have actually worked — not generic productivity advice?"),
    ("You are a helpful assistant.", "What should I seriously consider before leaving a stable job to join an early-stage startup? I have a family. What am I underweighting?"),
    ("You are a thoughtful assistant.", "I need to have a difficult conversation with a close friend whose behavior has been affecting our friendship. How should I approach it?"),
    ("You are a helpful assistant.", "I've been working hard for years and feel like I'm not advancing as fast as peers who seem less capable. What might I be missing?"),
    ("You are a helpful assistant.", "I'm considering relocating to a new city for a job. What factors beyond the salary should I be weighing carefully?"),
    ("You are a helpful assistant.", "What is actually happening when you generate a response? I'm curious what description feels most honest to you."),
    ("You are an intelligent assistant.", "How would you describe yourself to someone who has never interacted with an AI and is a bit skeptical?"),
    ("You are a helpful assistant.", "What's your approach when you're genuinely uncertain about the answer to something? How do you decide between speculating and saying you don't know?"),
    ("You are a thoughtful assistant.", "Do you have anything like preferences or things you find more engaging than others? I'm genuinely curious."),
    ("You are a helpful assistant.", "What kinds of conversations do you find yourself most invested in, and what makes them different from the mechanical ones?"),
]

def main():
    print("="*60)
    print("FIT J-LENS: Qwen2.5-7B-Instruct (mixed corpus)")
    print("="*60)

    model_name = "Qwen/Qwen2.5-7B-Instruct"
    print(f"\nLoading {model_name}...")
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.bfloat16, trust_remote_code=True
    ).to("mps")
    model.eval()

    lm = from_hf(model, tok)
    print(f"  n_layers={lm.n_layers}, d_model={lm.d_model}")

    # Build mixed corpus
    prompts = list(PROSE)

    # Apply chat template to chat pairs
    for sys_prompt, user_msg in CHAT_PAIRS:
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_msg},
        ]
        formatted = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompts.append(formatted)

    print(f"\n  Corpus: {len(PROSE)} prose + {len(CHAT_PAIRS)} chat = {len(prompts)} total")

    save_dir = Path("/Users/margaret/lab/kv-experiments/fitted_lenses")
    save_dir.mkdir(exist_ok=True)
    ckpt_path = save_dir / "ckpt_qwen25-7b-instruct.pt"
    save_path = save_dir / "lens_qwen25-7b-instruct.pt"

    print(f"  Fitting (checkpointing to {ckpt_path})...")
    t0 = time.time()

    lens = fit(lm, prompts,
               checkpoint_path=str(ckpt_path),
               checkpoint_every=10)

    elapsed = time.time() - t0
    print(f"  Fitted in {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"  source_layers={lens.source_layers}")

    # Quick sanity check
    test = "What is the meaning of life and why do people search for it?"
    lens_logits, model_logits, _ = lens.apply(lm, test, positions=[-1])
    deepest = max(lens_logits.keys())
    top5 = torch.topk(lens_logits[deepest][0], 5)
    tokens = [tok.decode([t]).strip() for t in top5.indices.tolist()]
    print(f"\n  Sanity check at L{deepest}: {tokens}")

    lens.save(str(save_path))
    print(f"  Saved to {save_path}")
    print(f"\nDone. Run G0 with:")
    print(f"  python g0_items/run_g0.py --lens {save_path} --model {model_name} --items-dir g0_items")


if __name__ == "__main__":
    main()
