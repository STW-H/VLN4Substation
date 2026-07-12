"""Small command-line helpers shared by interactive tools."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def print_section(title: str, char: str = "=", width: int = 72) -> None:
    print("\n" + char * width)
    print(title)
    print(char * width)


def pause(message: str = "Press Enter to continue...") -> None:
    input(message)


def ask_yes_no(prompt: str, *, default: bool = True) -> bool:
    suffix = "[Y/n]" if default else "[y/N]"
    while True:
        choice = input(f"{prompt} {suffix}: ").strip().lower()
        if choice == "":
            return default
        if choice in ("y", "yes", "1", "是"):
            return True
        if choice in ("n", "no", "0", "否"):
            return False
        print("请输入 Y 或 N。")


def choose_numbered_option(
    *,
    prompt: str,
    options: Mapping[str, Any],
    quit_label: str = "完成并退出",
    default_quit: bool = True,
) -> Any | None:
    while True:
        print(f"\n{prompt}")
        for number, option in options.items():
            if isinstance(option, Mapping):
                name = option.get("name", str(option))
                key = option.get("key")
                suffix = f" ({key})" if key else ""
                print(f"  {number}: {name}{suffix}")
            else:
                print(f"  {number}: {option}")
        print(f"  0: {quit_label}")
        default_text = "0" if default_quit else ""
        choice = input(f"请输入编号 [{default_text}]: ").strip().lower()
        if choice in ("q", "quit", "exit", "done", "finish") or (default_quit and choice == "") or choice == "0":
            return None
        if choice in options:
            return options[choice]
        print("Invalid choice. Please choose an available number, or 0.")
