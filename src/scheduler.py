"""
Schedules the execution of n threads in parallel and correctly writes their
responses atomically into an output JSONL file. Handles resuming after a crash.
"""

from copy import deepcopy
import json
import random
import threading
import os
from turtle import up
import uuid

from dataclasses import dataclass, field

from config_reader import config
from conversation_generation import generate_conversation, generate_initial_question
from llm_interface import clean_response, retrieve_several_as_structured_output
from prompts import (
    INITIAL_MESSAGE_PROMPT,
    SYSTEM_PROMPT_GENERATION_ADDITION,
    SYSTEM_PROMPT_GENERATION_PROMPT,
    CreateSpecialPrompts,
    concatenate_prompts,
)

from logging_manager import logger
from tools import pick_random_tools

_JSONL_WRITE_LOCK = threading.Lock()


@dataclass
class Category:
    """
    Stores meta information about a category.
    """

    n_rows_total: int = 0
    n_rows_per_language: dict[str, int] = field(default_factory=dict)
    name: str = ""

    lang_ptr: int = 0  # cycles through


class NoMoreLanguagesError(Exception):
    """
    Raised when there are no more languages left for a category.
    """


class NoMoreCategoriesError(Exception):
    """
    Raised when there are no more categories left with available languages.
    """


def next_lang(category: Category) -> str:
    """
    Returns the next language for the given category, based on the language
    distribution and the lang_ptr. Also updates the lang_ptr to point to the
    next language for the next call.
    """

    languages = list(category.n_rows_per_language.keys())
    n_languages = len(languages)

    def advance_lang_ptr():
        """
        Advances the lang_ptr to the next language.
        """

        category.lang_ptr = (category.lang_ptr + 1) % n_languages

    for _ in range(n_languages):
        if category.n_rows_per_language[languages[category.lang_ptr]] > 0:
            break

        advance_lang_ptr()
    else:
        raise NoMoreLanguagesError(f"No languages left for category {category.name}")

    next_language = languages[category.lang_ptr]
    category.n_rows_per_language[next_language] -= 1
    advance_lang_ptr()

    return next_language


def get_specials(special_categories: dict) -> str:
    """
    Returns a list of special conditions to include in the prompt, based on the
    special_categories and their percentages.
    """

    map_ = {
        "prompt_injection_refusals": "prompt_injection",
        "hallucination_rejections": "hallucination",
        "pushback_on_nonsense_questions": "nonsense",
    }

    weights = {
        "none": 1.0,
        "prompt_injection": 0.0,
        "hallucination": 0.0,
        "nonsense": 0.0,
    }

    for key in special_categories:
        assert key in map_, f"Unknown special category {key}"

        specific_row = special_categories.get(
            key, {"activate": False, "percentage": 0.0}
        )

        if not specific_row["activate"]:
            continue

        weights[map_[key]] = specific_row["percentage"]
        weights["none"] -= specific_row["percentage"]

    if weights["none"] < 0:
        raise ValueError("The sum of special category percentages cannot exceed 1.0")

    choices, probabilities = zip(*weights.items())
    category = random.choices(choices, probabilities, k=1)[0]

    return "" if category == "none" else category


def initialize_output_directory(run_name: str) -> tuple[str, str, str]:
    """
    Initializes the output directory for the given run name.
    """

    RUN_NAME = run_name + "_" + str(uuid.uuid4())[:8]

    os.makedirs(os.path.join("data", RUN_NAME), exist_ok=True)
    logger.debug("Initialized output directory: %s", RUN_NAME)

    # create system_prompts.jsonl, conversations.jsonl and state.json files
    system_prompts_path = os.path.join("data", RUN_NAME, f"system_prompts.jsonl")
    conversations_path = os.path.join("data", RUN_NAME, f"conversations.jsonl")
    state_path = os.path.join("data", RUN_NAME, f"state.json")

    with open(system_prompts_path, "w") as f:
        pass

    with open(conversations_path, "w") as f:
        pass

    with open(state_path, "w") as f:
        json.dump(
            {
                "run_name": RUN_NAME,
                "n_rows_total": config["rows"],
                "categories": config["categories"],
                "languages": config["languages"],
                "templates": [],
            },
            f,
            indent=4,
        )

    return system_prompts_path, conversations_path, state_path


def update_state_file(state_file_path: str, diff: dict) -> None:
    """
    Updates the state file with the given differences.
    """

    with open(state_file_path, "r") as f:
        state = json.load(f)

    state.update(diff)

    with open(state_file_path, "w") as f:
        json.dump(state, f, indent=4)


def read_state_file(state_file_path: str) -> dict:
    """
    Reads the state file and returns its content as a dictionary.
    """

    with open(state_file_path, "r") as f:
        state = json.load(f)

    return state


def write_atomically_to_jsonl(file_path: str, data: list[dict]) -> None:
    """
    Using the threading library, writes the given data atomically to the given
    file path in JSONL format.
    """

    with _JSONL_WRITE_LOCK:
        logger.debug("Writing %d items to %s", len(data), file_path)
        with open(file_path, "a") as file_handle:
            file_handle.write(
                "\n".join([json.dumps(item, ensure_ascii=False) for item in data])
            )
            if data:
                file_handle.write("\n")
        logger.debug("Wrote %d items to %s", len(data), file_path)


def generate_system_prompts_in_parallel(output_file_path: str) -> None:
    """
    Generates system prompts in parallel using multiple threads. Needs to store
    the generated system prompts in the output_file_path atomically across
    threads.
    """

    themes = config["prompting"]["system_prompt_generation_themes"]
    formats = config["prompting"]["system_prompt_formats"]
    n_system_prompts = config["prompting"]["n_system_prompts"]
    n_threads = config["n_threads"]

    assert isinstance(n_system_prompts, int) and n_system_prompts > 0
    assert isinstance(n_threads, int) and n_threads > 0

    def worker(n_prompts: int):
        """
        The worker function for generating system prompts. Each thread runs
        this function with its own theme and format.
        """

        is_special = (
            config["prompting"]["include_special_system_prompts"]
            and random.random()
            < config["prompting"]["special_system_prompts_percentage"]
        )

        left_to_generate = n_prompts

        logger.debug(
            "Thread %s: starting to generate %d prompts (special=%s)",
            threading.get_ident(),
            n_prompts,
            is_special,
        )

        while left_to_generate > 0:
            theme = random.choice(themes)
            format_ = random.choice(formats)

            base_prompt = SYSTEM_PROMPT_GENERATION_PROMPT % (
                theme,
                format_,
                min(left_to_generate, config["batch_size"]),
                config["prompting"]["system_prompt_extra_info"]
            )
            input_prompt = (
                base_prompt
                if not is_special
                else concatenate_prompts(base_prompt, SYSTEM_PROMPT_GENERATION_ADDITION)
            )

            prompts = retrieve_several_as_structured_output(
                input_prompt,
                resp_json_array_name="prompts",
            )

            if not prompts:
                logger.error(
                    "Thread %s: Failed to generate system prompts.",
                    threading.get_ident(),
                )
                raise ValueError("Failed to generate system prompts.")

            prompts_to_write = [clean_response(prompt) for prompt in prompts]
            left_to_generate -= len(prompts_to_write)

            # write prompts atomically to file:
            write_atomically_to_jsonl(
                output_file_path,
                [{"prompt": prompt} for prompt in prompts_to_write],
            )

        logger.debug("Thread %s: finished generating prompts", threading.get_ident())

    if n_system_prompts == 0:
        logger.info("No system prompts requested (n_system_prompts == 0)")
        return

    thread_count = min(n_threads, n_system_prompts)
    base_prompts_per_thread, remainder = divmod(n_system_prompts, thread_count)

    threads = []

    for thread_index in range(thread_count):
        prompts_for_thread = base_prompts_per_thread
        if thread_index < remainder:
            prompts_for_thread += 1

        thread = threading.Thread(target=worker, args=(prompts_for_thread,))
        thread.start()
        logger.debug(
            "Started system prompt thread %d (id=%s) with %d prompts",
            thread_index,
            thread.ident,
            prompts_for_thread,
        )
        threads.append(thread)

    for thread in threads:
        thread.join()

    logger.info("All system prompt threads joined")


def generate_first_user_messages_in_parallel(templates: list[dict[str, str]]) -> None:
    """
    Populates all of the template entries with a generated first user message.
    Uses batching to make the process more efficient and creative.
    """

    # will only store the references to the templates, so this structure is
    # actually very light-weight
    slots: dict[int, list[dict[str, str]]] = {}

    # group templates by category and language
    for template in templates:
        category = template["category"]
        language = template["language"]
        specials = template["specials"]

        category_id = hash(category + language + specials)
        modifier = 0

        while category_id in slots and len(slots[category_id]) >= config["batch_size"]:
            modifier += 1
            category_id = hash(category + language + specials + str(modifier))

        if category_id not in slots:
            slots[category_id] = []

        slots[category_id].append(template)

    def worker(jobs: list[list[dict[str, str]]]):
        for job in jobs:
            # as we know that the templates in the same job have the same
            # category and language, we can generate n initial messages where n
            # is the length of the job
            category = job[0]["category"]
            language = job[0]["language"]
            specials = job[0]["specials"]
            to_generate = len(job)

            backstage_user = config["prompting"].get("backstage_user", "").strip()

            if not specials:
                input_prompt = INITIAL_MESSAGE_PROMPT % (
                    category,
                    language,
                    to_generate,
                    backstage_user
                )
            else:
                c = CreateSpecialPrompts()

                input_prompt = {
                    "nonsense": c.nonsense_prompt,
                    "prompt_injection": c.prompt_injection_prompt,
                    "hallucination": c.get_hallucination_prompt(),
                }[specials] % (category, language, to_generate, backstage_user)

            try:
                messages = retrieve_several_as_structured_output(
                    input_prompt,
                    resp_json_array_name="first_user_messages",
                )
            except Exception as e:
                logger.exception(
                    "Thread %s: Failed to generate first user messages for job %s: %s",
                    threading.get_ident(),
                    job,
                    e,
                )
                continue

            if len(messages) < to_generate:
                logger.warning(
                    "Thread %s: Generated fewer messages than needed (%d < %d), reusing some prompts.",
                    threading.get_ident(),
                    len(messages),
                    to_generate,
                )

                while len(messages) < to_generate:
                    messages.extend(messages)

            elif len(messages) > to_generate:
                logger.warning(
                    "Thread %s: Generated more messages than needed (%d > %d), cutting off the extras.",
                    threading.get_ident(),
                    len(messages),
                    to_generate,
                )
                messages = messages[:to_generate]

            if not messages:
                logger.error(
                    "Thread %s: Failed to generate first user messages.",
                    threading.get_ident(),
                )
                continue

            messages = [clean_response(prompt) for prompt in messages]
            random.shuffle(messages)
        
            for template, message in zip(job, messages):
                template["initial_message"] = message

    threads = []
    thread_count = min(config["n_threads"], len(slots))
    slots_per_thread = len(slots) // thread_count

    for thread_index in range(thread_count):
        assigned_slots = list(slots.values())[
            thread_index * slots_per_thread : (thread_index + 1) * slots_per_thread
        ]

        # assign all of the remaining slots to the first thread as well
        if thread_index == 0:
            remaining_slots = list(slots.values())[thread_count * slots_per_thread :]
            assigned_slots.extend(remaining_slots)

        thread = threading.Thread(target=worker, args=(assigned_slots,))
        thread.start()
        logger.debug(
            "Started first user message generation thread %d (id=%s) with %d slots",
            thread_index,
            thread.ident,
            len(assigned_slots),
        )

        threads.append(thread)
    
    for thread in threads:
        thread.join()
    
    logger.debug("Finished generating first user messages for all templates")


def generate_templates(
    state_path: str,
    system_prompts_path: str,
    distribution: list[Category],
    config: dict,
) -> int:
    """
    Generates templates for later generation using parallel workers; for now,
    this only means generating the conditions (category, language, system
    prompt) for each conversation to be generated, and storing them in the
    state file. This makes resuming later easier.

    Returns the actual number of templates generated.
    """

    with open(system_prompts_path, "r") as f:
        system_prompts = [json.loads(line)["prompt"] for line in f]

    leftover_distribution = deepcopy(distribution)
    category_ptr = 0
    system_prompt_ptr = 0

    random.shuffle(system_prompts)
    random.shuffle(leftover_distribution)

    print(distribution)

    def get_template():
        nonlocal category_ptr, system_prompt_ptr

        for _ in range(len(leftover_distribution)):
            chosen_category = leftover_distribution[category_ptr]
            has_system_prompt = random.random() < config["prompting"]["system_prompt_probability"]

            try:
                conditions = {
                    "id": str(uuid.uuid4())[:8],
                    "category": chosen_category.name,
                    "language": next_lang(leftover_distribution[category_ptr]),
                    "system_prompt": system_prompts[system_prompt_ptr] if has_system_prompt else "",
                    "specials": get_specials(config["special_categories"]),
                    "initial_message": "", # will be filled later
                }
                
                if has_system_prompt:
                    system_prompt_ptr = (system_prompt_ptr + 1) % len(system_prompts)

                return conditions
            except NoMoreLanguagesError:
                category_ptr = (category_ptr + 1) % len(leftover_distribution)
        else:
            raise NoMoreCategoriesError("No categories left with available languages.")

    templates: list[dict[str, str]] = []

    while True:
        try:
            template = get_template()
            templates.append(template)
        except NoMoreCategoriesError:
            break

    logger.debug(
        "Generated %d templates (aimed for %d), writing state",
        len(templates),
        config["rows"],
    )
    update_state_file(state_path, {"templates": templates})

    logger.debug(
        "Grouping by category and language and generating first user message "
        "for each template"
    )

    generate_first_user_messages_in_parallel(templates)

    logger.debug("Finished generating templates with first user messages")
    update_state_file(state_path, {"templates": templates})

    return len(templates)


def generate_conversations_in_parallel(
    state_path: str,
    conversations_path: str,
    config: dict,
) -> None:
    """
    Generates conversations in parallel using multiple threads. Re-uses the
    previously generated system prompts from the system_prompts_path. Stores
    the generated conversations in the output_file_path atomically across
    threads.
    """

    state = read_state_file(state_path)
    templates = state["templates"]

    with open(conversations_path, "r") as f:
        already_generated_conversations = [json.loads(line) for line in f]

    already_generated_ids = set(conv["id"] for conv in already_generated_conversations)
    left_to_do = list(
        filter(lambda template: template["id"] not in already_generated_ids, templates)
    )

    def worker(templates_to_process: list[dict[str, str]]):
        """
        The worker function for generating conversations. Each thread runs this
        function with its own templates to process.
        """

        generated_conversations = []

        for template in templates_to_process:
            try:
                conversation = generate_conversation(
                    id_=template["id"],
                    category=template["category"],
                    system_prompt=template["system_prompt"],
                    special_category=template["specials"],
                    initial_question=template["initial_message"],
                    language=template["language"],
                    tools=pick_random_tools(),
                )

                generated_conversations.append(conversation)
            except Exception as e:
                logger.exception(
                    "Failed to generate conversation for template %s: %s",
                    template,
                    e,
                )

            if len(generated_conversations) >= config["write_every_n_conversations"]:
                # write conversations atomically to file:
                write_atomically_to_jsonl(conversations_path, generated_conversations)
                generated_conversations = []

        # write conversations atomically to file:
        write_atomically_to_jsonl(conversations_path, generated_conversations)

    n_threads = config["n_threads"]
    thread_count = min(n_threads, len(left_to_do))
    base_templates_per_thread, remainder = divmod(len(left_to_do), thread_count)

    threads = []

    for thread_index in range(thread_count):
        templates_for_thread = left_to_do[
            thread_index
            * base_templates_per_thread : (thread_index + 1)
            * base_templates_per_thread
        ]

        if thread_index < remainder:
            templates_for_thread.append(
                left_to_do[(thread_index + 1) * base_templates_per_thread]
            )

        thread = threading.Thread(target=worker, args=(templates_for_thread,))
        thread.start()

        logger.debug(
            "Started conversation generation thread %d (id=%s) with %d templates",
            thread_index,
            thread.ident,
            len(templates_for_thread),
        )

        threads.append(thread)

    for thread in threads:
        thread.join()


def calculate_per_language(
    total_rows: int, languages: dict[str, float]
) -> tuple[dict[str, int], int]:
    """
    Calculates the number of rows for each language based on their percentages.
    Returns the number of rows for each language and the minimum total number
    of rows across all languages.

    Will at least allocate config["batch_size"] rows for each language to
    ensure that the batching works efficiently, which may inflate the total
    number of rows beyond the initially requested total_rows.
    """

    results = {}
    total_percent = sum(percent for percent in languages.values())

    for lang, percent in languages.items():
        results[lang] = max(
            int(total_rows * percent / total_percent), config["batch_size"]
        )

    min_total_rows = sum(results.values())

    return results, min_total_rows


def distribute_categories(
    total_rows: int, categories: dict[str, float | int]
) -> list[Category]:
    """
    Distributes the total number of rows across different categories based on
    their weights.
    """

    results = []

    rows_left = total_rows

    # first, handle all categories with a fixed number of rows:
    for category_name, weight in categories.items():
        if isinstance(weight, int) and weight >= 0:
            language_distribution, total_rows_for_category = calculate_per_language(
                weight, config["languages"]
            )

            results.append(
                Category(
                    n_rows_total=total_rows_for_category,
                    n_rows_per_language=language_distribution,
                    name=category_name,
                )
            )

            # we do not subtract by total_rows_for_category here, because the
            # user shall not need to care about the batch_size when writing the
            # config
            rows_left -= weight

        assert (
            rows_left >= 0
        ), "Total number of rows is less than the sum of fixed rows."


    # now, let's see how many percent the remaining categories have in total:
    total_percent = sum(
        weight for weight in categories.values() if isinstance(weight, float)
    )

    # let's count how many categories are -1 ("go figure it out for me"):
    n_negative_one = sum(1 for weight in categories.values() if weight == -1)
    all_used = total_percent == 1.0 or n_negative_one > 0

    # we do this in order to evenly distribute the -1 weights across the
    # remaining categories:
    for category_name, weight in categories.items():
        if weight == -1:
            categories[category_name] = (1 - total_percent) / n_negative_one

    assert all_used, "The category distribution does not use up all of the \
requested rows. Have you checked that your percentages add up to 1.0?"

    # now we can distribute the remaining rows according to the percentage
    # weights:
    for category_name, weight in categories.items():
        if isinstance(weight, float):
            n_rows_for_category = int(total_rows * weight)

            language_distribution, actual_rows_for_category = calculate_per_language(
                n_rows_for_category, config["languages"]
            )

            results.append(
                Category(
                    n_rows_total=actual_rows_for_category,
                    n_rows_per_language=language_distribution,
                    name=category_name,
                )
            )

    return results


def main_flow(
    run_name: str = "",
    new_run: bool = True,
    resume_run: bool = False,
    generate_templates_only: bool = False,
):
    """
    Main flow of the program. Needs to handle a lot: Orchestrating the topic,
    language and system prompt distribution among conversations, correctly
    batching the generated initial prompts and system prompts etc.
    """

    assert not (new_run and resume_run)
    assert (
        generate_templates_only and not (resume_run or new_run)
    ) or not generate_templates_only

    if not run_name:
        run_name = config["run_name"]

    n_threads = config["n_threads"]
    n_rows = config["rows"]

    logger.debug("Main flow started: n_threads=%d, n_rows=%d", n_threads, n_rows)

    assert isinstance(n_threads, int) and n_threads > 0
    assert isinstance(n_rows, int) and n_rows > 0

    if not resume_run:
        system_prompts_path, conversations_path, state_path = initialize_output_directory(
            config["run_name"]
        )
    else:
        system_prompts_path = os.path.join("data", run_name, f"system_prompts.jsonl")
        conversations_path = os.path.join("data", run_name, f"conversations.jsonl")
        state_path = os.path.join("data", run_name, f"state.json")

    if resume_run:
        state = read_state_file(state_path)

        with open(conversations_path, "r") as f:
            n_generated = len([json.loads(line) for line in f])

        logger.debug(
            "Resuming run %s: %d/%d rows generated",
            state["run_name"],
            n_generated,
            state["n_rows_total"],
        )

        generate_conversations_in_parallel(state_path, conversations_path, config)
        return

    distribution = distribute_categories(n_rows, config["categories"])

    generate_system_prompts_in_parallel(system_prompts_path)
    n_templates = generate_templates(state_path, system_prompts_path, distribution, config)

    if not generate_templates_only:
        generate_conversations_in_parallel(state_path, conversations_path, config)
    else:
        logger.info(
            f"Templates generated (n={n_templates}), you can resume this run "
            "later by starting simple-sft with --resume-run and the same run "
            f"name: {run_name}")
