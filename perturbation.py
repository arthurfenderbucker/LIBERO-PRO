import os
import re
import random
import yaml
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import subprocess


# -----------------------------
# Existing classes: BDDLParser / 5 perturbators
# (Keep implementation unchanged, reuse your code above)
# -----------------------------

class BDDLParser:
    """Parse BDDL files and extract relevant information"""

    def __init__(self, file_content: str):
        self.file_content = file_content
        self.objects_of_interest = self._parse_obj_of_interest()
        self.initial_states = self._parse_initial_states()

    def _parse_obj_of_interest(self) -> List[str]:
        """Parse objects of interest"""
        obj_pattern = r'\(:obj_of_interest(.*?)\)'
        obj_match = re.search(obj_pattern, self.file_content, re.DOTALL)
        if not obj_match:
            return []
        obj_content = obj_match.group(1)
        objects = re.findall(r'(\w+_\d+)', obj_content)
        return objects

    def _parse_initial_states(self) -> Dict[str, str]:
        """
        Parse bddl (:init ...) section, return initial_states[obj] = region
        """
        initial_states = {}
        init_block_match = re.search(r"\(:init(.*?)(?=\)\s*\(:goal|\)\s*$)", self.file_content, re.S)
        if not init_block_match:
            return initial_states
        init_text = init_block_match.group(1)
        for match in re.finditer(r"\(On\s+(\w+)\s+(\w+)\)", init_text):
            obj, region = match.groups()
            initial_states[obj] = region
        return initial_states


class SwapPerturbator:
    """Perform swap perturbation based on configuration file"""

    def __init__(self, parser: BDDLParser, config_path: str):
        self.parser = parser
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

    def perturb(self, task_suite_name: str, task_name: str) -> str:
        content = self.parser.file_content
        objs_interest = list(self.parser.objects_of_interest or [])
        if not objs_interest:
            print("No objects of interest found")
            return content

        task_cfg = self.config.get(task_suite_name, {}).get(task_name, None)
        if task_cfg is None:
            print(f"Task {task_name} has no allowed_swaps configuration")
            return content

        init_states = dict(self.parser.initial_states)

        used = set()
        pairs = []

        random.shuffle(objs_interest)

        def candidates_for(obji: str):
            if isinstance(task_cfg, dict):
                if obji in task_cfg and isinstance(task_cfg[obji], list):
                    return list(task_cfg[obji])
                if "__any__" in task_cfg and isinstance(task_cfg["__any__"], list):
                    return list(task_cfg["__any__"])
                return []
            elif isinstance(task_cfg, list):
                return list(task_cfg)
            else:
                return []

        for obj in objs_interest:
            if obj in used:
                continue
            cand_pool = candidates_for(obj)
            cand_pool = [
                x for x in cand_pool
                if x != obj and x in init_states and x not in used
            ]
            if not cand_pool:
                print(f"[Skip] Object of interest {obj} has no available candidates (may not be in init or already used)")
                continue

            swap_obj = random.choice(cand_pool)

            reg_a = init_states.get(obj)
            reg_b = init_states.get(swap_obj)
            if not reg_a or not reg_b:
                print(f"[Skip] {obj} or {swap_obj} not in init, cannot swap")
                continue

            pat_a = rf"\(On\s+{re.escape(obj)}\s+{re.escape(reg_a)}\s*\)"
            pat_b = rf"\(On\s+{re.escape(swap_obj)}\s+{re.escape(reg_b)}\s*\)"

            content, n1 = re.subn(pat_a, f"(On {obj} {reg_b})", content, count=1)
            content, n2 = re.subn(pat_b, f"(On {swap_obj} {reg_a})", content, count=1)

            if n1 > 0 and n2 > 0:
                init_states[obj], init_states[swap_obj] = reg_b, reg_a
                used.add(obj)
                used.add(swap_obj)
                pairs.append((obj, swap_obj))
                print(f"Task {task_name}: Swapped positions of {obj} and {swap_obj}")
            else:
                print(f"[Warning] On statement for {obj} or {swap_obj} not matched, BDDL format may not match regex")

        if not pairs:
            print("No swap pairs formed, file not modified")

        return content


class ObjectReplacePerturbator:
    """
    Perform "object replacement perturbation" based on ood_object.yaml
    """

    def __init__(self, parser: BDDLParser, config_path: str):
        self.parser = parser
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

    def _extract_language_span(self, text: str):
        m = re.search(r"\(:language\b.*?\)", text, flags=re.S)
        return m.span() if m else None

    def perturb(self, task_suite_name: str, task_name: str, seed: Optional[int] = None) -> str:
        if seed is not None:
            random.seed(seed)

        suite_cfg = self.config.get(task_suite_name, {})
        task_cfg: Dict[str, List[str]] = suite_cfg.get(task_name, {})

        if not task_cfg:
            print(f"[Object Replace] Task {task_name} has no entry in configuration, skipping.")
            return self.parser.file_content

        mapping: Dict[str, str] = {}
        for obj_interest, candidates in task_cfg.items():
            if not candidates:
                print(f"[Object Replace] {obj_interest} has no replacement candidates, skipping.")
                continue
            chosen = random.choice(candidates)
            mapping[obj_interest] = chosen

        if not mapping:
            print("[Object Replace] No replacement mapping formed, skipping.")
            return self.parser.file_content

        content = self.parser.file_content
        lang_span = self._extract_language_span(content)
        if lang_span:
            s, e = lang_span
            prefix = content[:s]
            language_block = content[s:e]
            suffix = content[e:]
        else:
            prefix, language_block, suffix = content, "", ""

        for old_name, new_name in mapping.items():
            prefix = prefix.replace(old_name, new_name)
            suffix = suffix.replace(old_name, new_name)

        new_content = prefix + language_block + suffix

        for k, v in mapping.items():
            print(f"[Object Replace] {task_name}: {k} -> {v}")

        return new_content


class LanguagePerturbator:
    """
    Read candidate instruction text from ood_language.yaml, randomly select one, and replace (:language ...)
    """

    def __init__(self, parser: BDDLParser, config_path: str):
        self.parser = parser
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f) or {}

    def _find_language_block(self, text: str):
        m = re.search(r"\(:language\s*(.*?)\)", text, flags=re.S)
        if not m:
            return None
        inner = m.group(1)
        inner_start = m.start(1)
        inner_end = m.end(1)
        return (inner_start, inner_end, inner)

    def perturb(self, task_suite_name: str, task_name: str, seed: Optional[int] = None) -> str:
        if seed is not None:
            random.seed(seed)

        candidates = (self.config.get(task_suite_name, {}) or {}).get(task_name, [])
        if not candidates:
            print(f"[Language Perturbation] Task {task_name} has no candidates in configuration, skipping.")
            return self.parser.file_content

        new_lang = random.choice(candidates)

        block = self._find_language_block(self.parser.file_content)
        if not block:
            print("[Language Perturbation] (:language ...) section not found, skipping.")
            return self.parser.file_content

        s, e, old_inner = block
        new_content = self.parser.file_content[:s] + new_lang + self.parser.file_content[e:]

        print(f"[Language Perturbation] {task_name}: '{old_inner}' -> '{new_lang}'")
        return new_content


class TaskPerturbator:
    """
    Read candidate tasks from ood_task.yaml, replace both (:language ...) and (:goal ...), and replace (:obj_of_interest ...)
    """

    def __init__(self, parser: BDDLParser, config_path: str):
        self.parser = parser
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f) or {}

    def _find_language_inner_span(self, text: str):
        m = re.search(r"\(:language\s*(.*?)\)", text, flags=re.S)
        if not m:
            return None
        return (m.start(1), m.end(1), m.group(1))

    def _find_outer_block_span(self, text: str, head: str):
        start = text.find(head)
        if start < 0:
            return None
        i, depth = start, 0
        while i < len(text):
            ch = text[i]
            if ch == '(':
                depth += 1
            elif ch == ')':
                depth -= 1
                if depth == 0:
                    return (start, i + 1)
            i += 1
        return None

    def _replace_language(self, text: str, new_lang: str) -> str:
        span = self._find_language_inner_span(text)
        if not span:
            print("[TaskPerturbator] (:language ...) section not found, skipping language replacement.")
            return text
        s, e, old = span
        print(f"[TaskPerturbator] language: '{old}' -> '{new_lang}'")
        return text[:s] + new_lang + text[e:]

    def _replace_goal(self, text: str, new_goal_expr: str) -> str:
        span = self._find_outer_block_span(text, "(:goal")
        if not span:
            print("[TaskPerturbator] (:goal ...) section not found, skipping goal replacement.")
            return text
        start, end = span
        replacement = "(:goal\n  " + new_goal_expr + "\n)"
        print(f"[TaskPerturbator] goal: replaced with {new_goal_expr}")
        return text[:start] + replacement + text[end:]

    def _replace_obj_of_interest(self, text: str, new_objs: list) -> str:
        span = self._find_outer_block_span(text, "(:obj_of_interest")
        if not span:
            print("[TaskPerturbator] (:obj_of_interest ...) section not found, skipping replacement.")
            return text
        start, end = span
        replacement = "(:obj_of_interest\n"
        for obj in new_objs:
            replacement += f"  {obj}\n"
        replacement += ")"
        print(f"[TaskPerturbator] obj_of_interest: replaced with {new_objs}")
        return text[:start] + replacement + text[end:]

    def perturb(self, task_suite_name: str, task_name: str, seed: Optional[int] = None) -> str:
        if seed is not None:
            random.seed(seed)

        suite_cfg = self.config.get(task_suite_name, {})
        task_cfg = suite_cfg.get(task_name, {})
        if not task_cfg:
            print(f"[TaskPerturbator] Task {task_name} has no candidates in configuration, skipping.")
            return self.parser.file_content

        language_options = list(task_cfg.keys())
        chosen_lang = random.choice(language_options)
        chosen_cfg = task_cfg.get(chosen_lang, {})
        chosen_goal = chosen_cfg.get("goal")
        chosen_objs = chosen_cfg.get("obj_of_interest", [])

        if not chosen_goal:
            print(f"[TaskPerturbator] Task {task_name}'s language '{chosen_lang}' has no goal, skipping.")
            return self.parser.file_content

        new_content = self.parser.file_content
        new_content = self._replace_language(new_content, chosen_lang)
        new_content = self._replace_goal(new_content, chosen_goal)
        new_content = self._replace_obj_of_interest(new_content, chosen_objs)
        return new_content


class EnvironmentReplacePerturbator:
    """
    Environment replacement perturbation (global direct replacement) + modify problem name scene marker + fix (:fixtures) right-side type
    """

    def __init__(self, parser: BDDLParser, config_path: str):
        self.ALLOWED_ENVS = {"main_table", "kitchen_table", "living_room_table", "study_table", "floor"}
        self.ENV_TOKEN = {
            "main_table": "Tabletop",
            "kitchen_table": "Kitchen_Tabletop",
            "living_room_table": "Living_Room_Tabletop",
            "study_table": "Study_Tabletop",
            "floor": "Floor",
        }
        self.ENV_FIXTYPE = {
            "main_table": "table",
            "kitchen_table": "kitchen_table",
            "living_room_table": "living_room_table",
            "study_table": "study_table",
            "floor": "floor",
        }
        self.parser = parser
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f) or {}

    def _extract_current_env(self, task_suite_name: str, task_name: str) -> Optional[str]:
        suite_cfg = self.config.get(task_suite_name, {})
        entry = suite_cfg.get(task_name)
        if entry is None:
            return None
        if isinstance(entry, list):
            return entry[0] if entry else None
        if isinstance(entry, str):
            return entry
        return None

    def _rewrite_problem_env_token(self, text: str, env_name: str) -> str:
        token = self.ENV_TOKEN.get(env_name, "Tabletop")
        pattern = r"\(define\s*\(problem\s+LIBERO_[A-Za-z_]*\)"
        replacement = f"(define (problem LIBERO_{token}_Manipulation)"
        return re.sub(pattern, replacement, text, count=1)

    def _find_outer_block_span(self, text: str, head: str):
        start = text.find(head)
        if start < 0:
            return None
        i, depth = start, 0
        while i < len(text):
            ch = text[i]
            if ch == '(':
                depth += 1
            elif ch == ')':
                depth -= 1
                if depth == 0:
                    return (start, i + 1)
            i += 1
        return None

    def _rewrite_fixtures_type(self, text: str, fixture_name: str, new_type: str) -> str:
        span = self._find_outer_block_span(text, "(:fixtures")
        if not span:
            return text
        s, e = span
        block = text[s:e]
        pattern = rf"(^\s*{re.escape(fixture_name)}\s*-\s*)([A-Za-z_][A-Za-z0-9_]*)"
        new_block, n = re.subn(pattern, rf"\1{new_type}", block, count=1, flags=re.M)
        if n == 0:
            return text
        return text[:s] + new_block + text[e:]

    def perturb(self, task_suite_name: str, task_name: str, seed: Optional[int] = None) -> str:
        if seed is not None:
            random.seed(seed)

        current_env = self._extract_current_env(task_suite_name, task_name)
        if not current_env:
            print(f"[Environment Replace] Task {task_name} environment not found in configuration (or empty list), skipping.")
            return self.parser.file_content
        if current_env not in self.ALLOWED_ENVS:
            print(f"[Environment Replace] Configured environment '{current_env}' not in allowed set {self.ALLOWED_ENVS}, skipping.")
            return self.parser.file_content

        candidates = list(self.ALLOWED_ENVS - {current_env})
        if not candidates:
            print("[Environment Replace] No candidate replacement environments, skipping.")
            return self.parser.file_content

        # new_env = random.choice(candidates)
        new_env = "living_room_table"
        new_content = self.parser.file_content.replace(current_env, new_env)
        new_content = self._rewrite_problem_env_token(new_content, new_env)
        new_fix_type = self.ENV_FIXTYPE.get(new_env, None)
        if new_fix_type:
            new_content = self._rewrite_fixtures_type(new_content, new_env, new_fix_type)

        print(f"[Environment Replace] {task_name}: {current_env} -> {new_env}")
        return new_content


# -----------------------------------------
# New: Combined perturbator (mixed execution by boolean switches)
# -----------------------------------------

@dataclass
class PerturbFlags:
    use_environment: bool = False
    use_swap: bool = False
    use_object: bool = False
    use_language: bool = False
    use_task: bool = False


class BDDLCombinedPerturbator:
    """
    Combined Perturbator:
    - Specify which perturbations to enable using PerturbFlags.
    - Specify the YAML path for each perturbation using configs.
      configs = {
        "environment": "./ood_environment.yaml",
        "swap": "./ood_spatial_relation.yaml",
        "object": "./ood_object.yaml",
        "language": "./ood_language.yaml",
        "task": "./ood_task.yaml",
      }
    - Default execution order:
        environment -> object -> language -> task
      Rules:
        1) If use_swap=True, then SwapPerturbator must execute first.
        2) If use_task=True, all other perturbations must be set to False.
    """

    def __init__(self, configs: Dict[str, str]):
        self.configs = configs or {}

    @staticmethod
    def _task_name_from_path(input_path: str) -> str:
        return os.path.basename(input_path).replace(".bddl", "").strip()

    @staticmethod
    def _apply_and_reparse(content: str, perturbator_cls, cfg_path: str,
                           call_kwargs: Dict[str, Any],
                           task_suite_name: str, task_name: str) -> str:
        """
        Construct parser and perturbator with current content, execute one perturbation; return new content.
        """
        parser = BDDLParser(content)
        perturbator = perturbator_cls(parser, cfg_path)
        new_content = perturbator.perturb(task_suite_name=task_suite_name, task_name=task_name, **call_kwargs)
        return new_content

    def perturb_content(self,
                        content: str,
                        task_suite_name: str,
                        task_name: str,
                        flags: PerturbFlags,
                        seed: Optional[int] = None) -> str:
        current = content

        # Rule checking
        # use_task mode must be mutually exclusive
        if flags.use_task:
            if flags.use_environment or flags.use_swap or flags.use_object or flags.use_language:
                raise ValueError("Other perturbations cannot be enabled when use_task=True!")

        # 1) If swap is enabled, it must be executed first
        if flags.use_swap:
            cfg = self.configs.get("swap")
            if cfg and os.path.exists(cfg):
                parser = BDDLParser(current)
                perturbator = SwapPerturbator(parser, cfg)
                current = perturbator.perturb(task_suite_name=task_suite_name, task_name=task_name)
            else:
                print("[Combined Perturbation] Missing swap configuration or path does not exist, skipping swap perturbation.")

        # 2) Other perturbations (executed in order)
        if flags.use_environment:
            cfg = self.configs.get("environment")
            if cfg and os.path.exists(cfg):
                current = self._apply_and_reparse(
                    current, EnvironmentReplacePerturbator, cfg,
                    {"seed": seed}, task_suite_name, task_name
                )
            else:
                print("[Combined Perturbation] Missing environment configuration or path does not exist, skipping environment replacement.")

        if flags.use_object:
            cfg = self.configs.get("object")
            if cfg and os.path.exists(cfg):
                current = self._apply_and_reparse(
                    current, ObjectReplacePerturbator, cfg,
                    {"seed": seed}, task_suite_name, task_name
                )
            else:
                print("[Combined Perturbation] Missing object configuration or path does not exist, skipping object replacement.")

        if flags.use_language:
            cfg = self.configs.get("language")
            if cfg and os.path.exists(cfg):
                current = self._apply_and_reparse(
                    current, LanguagePerturbator, cfg,
                    {"seed": seed}, task_suite_name, task_name
                )
            else:
                print("[Combined Perturbation] Missing language configuration or path does not exist, skipping language replacement.")

        # 3) Task perturbation (if enabled, and ensure others are disabled)
        if flags.use_task:
            cfg = self.configs.get("task")
            if cfg and os.path.exists(cfg):
                current = self._apply_and_reparse(
                    current, TaskPerturbator, cfg,
                    {"seed": seed}, task_suite_name, task_name
                )
            else:
                print("[Combined Perturbation] Missing task configuration or path does not exist, skipping task replacement.")

        return current


class EvalEnvCreator:
    def __init__(self, input_dir: str, base_output_dir: str = None, script_path: str = "generate_init_states.py"):
        """
        Initialize the evaluation environment creator.

        :param input_dir: Directory containing the input BDDL files, e.g.:
                  /LIBERO/libero/libero/bddl_files/libero_goal_temp
        :param base_output_dir: Base path for the output directory (optional).
                    If not provided, "bddl_files" in input_dir will be automatically replaced with "init_files".
        """
        self.input_dir = input_dir.rstrip("/")
        self.folder_name = os.path.basename(self.input_dir)
        self.script_path = script_path

        if base_output_dir:
            self.output_dir = os.path.join(base_output_dir, self.folder_name)
        else:
            # Automatically replace bddl_files → init_files
            self.output_dir = self.input_dir.replace("bddl_files", "init_files")

    def create_env(self):
        """
        Create evaluation environment and execute generate_init_states.py
        """
        os.makedirs(self.output_dir, exist_ok=True)

        cmd = [
            sys.executable, self.script_path,
            "--bddl_base_dir", self.input_dir,
            "--output_dir", self.output_dir
        ]

        print(f"[INFO] Running command: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)



# -----------------------------------------
# Convenient Method: Process a Single .bddl File (Read -> Apply Mixed Perturbations -> Write)
# -----------------------------------------

def process_bddl_file_mixed(input_dir: str,
                            task_suite_name: str,
                            flags: PerturbFlags,
                            configs: Dict[str, str],
                            seed: Optional[int] = None) -> None:
    """
        Apply perturbations to BDDL files in the specified directory and save them to a temporary directory.

        Args:
            input_dir (str): Directory containing the input BDDL files.
            configs (dict): Perturbator configurations.
            task_suite_name (str): Name of the task suite.
            flags (dict): Perturbation parameter flags.
            seed (int): Random seed.
        """
    input_path = Path(input_dir)
    output_dir = input_path.parent / f"{input_path.name}_temp"
    output_dir.mkdir(parents=True, exist_ok=True)

    for file_path in input_path.glob("*.bddl"):
        with file_path.open("r", encoding="utf-8") as f:
            content = f.read()

        task_name = file_path.stem  # Remove file extension from filename
        pipeline = BDDLCombinedPerturbator(configs=configs)
        new_content = pipeline.perturb_content(
            content=content,
            task_suite_name=task_suite_name,
            task_name=task_name,
            flags=flags,
            seed=seed
        )

        output_path = output_dir / file_path.name
        with output_path.open("w", encoding="utf-8") as f:
            f.write(new_content)

    print(f"[Combined Perturbation] Processing complete, output: {output_dir}")
    return str(output_dir)


# -----------------------------------------
# Example main: equivalent to original main, but using combined perturbation method
# -----------------------------------------

def create_env(
    configs: dict = None,
):
    """
    Create evaluation environment

    :param input_path: Input bddl file path
    :param script_path: Path to generate_init_states.py
    :param init_output_dir: Final init file output path
    :param task_suite_name: Task suite name (default: libero_goal)
    :param seed: Random seed (default: 28, None for completely random)
    :param flags: Perturbation switches (default: environment/swap/object/language enabled, task disabled)
    :param configs: Perturbation configuration file paths
    """
    # Default perturbation flags
    flags = PerturbFlags(
        use_environment=configs.get("use_environment", False),
        use_swap=configs.get("use_swap", False),
        use_object=configs.get("use_object", False),
        use_language=configs.get("use_language", False),
        use_task=configs.get("use_task", False),
    )

    ood_task_configs = configs.get("ood_task_configs", {})

    # Generate temporary bddl output path
    temp_output_dir = process_bddl_file_mixed(
        input_dir=configs.get("bddl_files_path", ""),
        task_suite_name=configs.get("task_suite_name", ""),
        flags=flags,
        configs=ood_task_configs,
        seed=configs.get("seed", int),
    )

    # Call EvalEnvCreator
    creator = EvalEnvCreator(
        input_dir=temp_output_dir,
        script_path=configs.get("script_path", ""),
        base_output_dir=configs.get("init_file_dir", ""),
    )
    creator.create_env()

    return



# if __name__ == '__main__':
#     create_env(
#         input_path="/LIBERO/libero/libero/bddl_files/libero_goal/",
#         script_path="/LIBERO/notebooks/generate_init_states.py",
#         init_output_dir="/LIBERO/libero/libero/init_files/",
#         task_suite_name="libero_goal",
#         seed=42,
#     )