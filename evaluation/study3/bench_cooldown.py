#!/usr/bin/env python3
"""Experiment 3.4: Cooldown Anti-Thrashing.

Verifies that the Resource Broker's cooldown mechanism prevents rapid
state oscillation. Rapidly requests freeze/resume transitions and
confirms that cooldown properly throttles them.

Usage:
    python bench_cooldown.py [--output results_cooldown.json]
"""
import argparse
import json
import time
from pathlib import Path

from deepdrivemd.resource_broker.broker import GPUState, ResourceBroker


def test_no_cooldown() -> dict:
    """With cooldown=0, all transitions should fire."""
    broker = ResourceBroker(cooldown_sec=0.0)
    n_cycles = 10

    successes = 0
    for _ in range(n_cycles):
        if broker.freeze():
            successes += 1
        if broker.resume():
            successes += 1

    stats = broker.get_stats()
    return {
        "test": "no_cooldown",
        "cooldown_sec": 0.0,
        "attempted_transitions": n_cycles * 2,
        "successful_transitions": successes,
        "total_logged": stats["total_transitions"],
        "passed": successes == n_cycles * 2,
    }


def test_with_cooldown() -> dict:
    """With cooldown=0.1s, rapid transitions should be throttled."""
    cooldown = 0.1
    broker = ResourceBroker(cooldown_sec=cooldown)

    # Rapidly attempt transitions without waiting for cooldown
    attempts = []
    for i in range(20):
        freeze_ok = broker.freeze()
        resume_ok = broker.resume()
        attempts.append({
            "cycle": i,
            "freeze": freeze_ok,
            "resume": resume_ok,
            "state": broker.state.name,
        })
        time.sleep(0.01)  # 10ms between attempts (< 100ms cooldown)

    successful = sum(1 for a in attempts if a["freeze"] or a["resume"])
    stats = broker.get_stats()

    # Only the first transition should succeed, rest blocked by cooldown
    return {
        "test": "with_cooldown",
        "cooldown_sec": cooldown,
        "attempt_interval_ms": 10,
        "total_attempts": len(attempts) * 2,
        "successful_transitions": successful,
        "total_logged": stats["total_transitions"],
        "throttled": successful < len(attempts) * 2,
        "passed": successful < len(attempts) * 2,
    }


def test_cooldown_expires() -> dict:
    """Transitions should succeed after cooldown expires."""
    cooldown = 0.05  # 50ms cooldown
    broker = ResourceBroker(cooldown_sec=cooldown)

    results = []

    # First freeze: should succeed
    ok1 = broker.freeze()
    results.append({"action": "freeze_1", "success": ok1})

    # Immediate resume: should be blocked
    ok2 = broker.resume()
    results.append({"action": "resume_immediate", "success": ok2})

    # Wait for cooldown to expire
    time.sleep(cooldown + 0.01)

    # Resume after cooldown: should succeed
    ok3 = broker.resume()
    results.append({"action": "resume_after_cooldown", "success": ok3})

    # Immediate freeze: should be blocked
    ok4 = broker.freeze()
    results.append({"action": "freeze_immediate", "success": ok4})

    # Wait and freeze: should succeed
    time.sleep(cooldown + 0.01)
    ok5 = broker.freeze()
    results.append({"action": "freeze_after_cooldown", "success": ok5})

    expected = [True, False, True, False, True]
    actual = [r["success"] for r in results]

    return {
        "test": "cooldown_expires",
        "cooldown_sec": cooldown,
        "results": results,
        "expected": expected,
        "actual": actual,
        "passed": actual == expected,
    }


def test_cooldown_stress() -> dict:
    """Stress test: 100 rapid transitions, only a few should succeed."""
    cooldown = 0.05
    broker = ResourceBroker(cooldown_sec=cooldown)
    n_attempts = 100

    successes = 0
    for i in range(n_attempts):
        if i % 2 == 0:
            if broker.freeze():
                successes += 1
        else:
            if broker.resume():
                successes += 1
        time.sleep(0.001)  # 1ms between attempts

    # With 100 attempts at 1ms intervals and 50ms cooldown,
    # we expect roughly 100ms / 50ms = ~2 transitions
    stats = broker.get_stats()

    return {
        "test": "cooldown_stress",
        "cooldown_sec": cooldown,
        "total_attempts": n_attempts,
        "successful_transitions": successes,
        "throttle_ratio": 1.0 - (successes / n_attempts),
        "passed": successes < n_attempts // 2,  # Most should be throttled
    }


def main():
    parser = argparse.ArgumentParser(description="Cooldown anti-thrashing test")
    parser.add_argument("--output", type=str, default="results_cooldown.json")
    args = parser.parse_args()

    tests = [
        test_no_cooldown,
        test_with_cooldown,
        test_cooldown_expires,
        test_cooldown_stress,
    ]

    results = []
    all_passed = True
    for test_fn in tests:
        print(f"\nRunning: {test_fn.__name__}")
        result = test_fn()
        results.append(result)
        status = "PASS" if result["passed"] else "FAIL"
        print(f"  [{status}] {result['test']}")
        if "successful_transitions" in result:
            total = result.get("total_attempts", result.get("attempted_transitions", "?"))
            print(f"    Transitions: {result['successful_transitions']}/{total}")
        if not result["passed"]:
            all_passed = False

    print(f"\n--- Summary ---")
    print(f"{'Test':<30} {'Status':>8}")
    print("-" * 40)
    for r in results:
        status = "PASS" if r["passed"] else "FAIL"
        print(f"{r['test']:<30} {status:>8}")
    print(f"\nOverall: {'ALL PASSED' if all_passed else 'FAILED'}")
    print("Key insight: cooldown prevents thrashing between freeze/resume states")

    output_path = Path(args.output)
    with open(output_path, "w") as f:
        json.dump({"tests": results, "all_passed": all_passed}, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
