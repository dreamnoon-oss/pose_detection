"""Post-processing: map recorded events to actions and validate sequence order."""


class SequenceAnalyzer:
    """Map timestamped rule-events to expected actions and check ordering."""

    def __init__(self, events, action_mapping, *, fps=1.0):
        self.events = events
        self.action_mapping = action_mapping
        self.fps = fps

    def analyze(self):
        """Run the analysis.

        Returns a dict with keys:
            actions:     list of per-action results (found, frame, side, angle, …)
            order_valid: bool — were detected actions in the expected order?
            all_found:   bool — were all expected actions detected?
            total_events: int
        """
        # Group events by rule name, preserving chronological order
        rule_events = {}
        for e in self.events:
            rule_events.setdefault(e['rule'], []).append(e)

        results = []
        for mapping in self.action_mapping:
            rule_name = mapping['rule']
            occurrence = mapping.get('occurrence', 1)
            candidates = rule_events.get(rule_name, [])

            if len(candidates) >= occurrence:
                ev = candidates[occurrence - 1]
                results.append({
                    'action': mapping['action'],
                    'rule': rule_name,
                    'occurrence': occurrence,
                    'frame': ev['frame'],
                    'timestamp': ev['frame'] / self.fps if self.fps else 0,
                    'side': ev['side'],
                    'angle': ev['angle'],
                    'found': True,
                })
            else:
                results.append({
                    'action': mapping['action'],
                    'rule': rule_name,
                    'occurrence': occurrence,
                    'found': False,
                })

        # Validate chronological order
        prev_frame = 0
        order_valid = True
        for r in results:
            if r['found']:
                if r['frame'] < prev_frame:
                    order_valid = False
                    break
                prev_frame = r['frame']

        return {
            'actions': results,
            'order_valid': order_valid,
            'all_found': all(r['found'] for r in results),
            'total_events': len(self.events),
        }

    def summary(self):
        """Return a human-readable summary string."""
        result = self.analyze()
        lines = []
        lines.append("=" * 50)
        lines.append("  动作序列分析")
        lines.append("=" * 50)

        for i, a in enumerate(result['actions'], 1):
            if a['found']:
                side_label = "左臂" if a['side'] == 'L' else \
                             "右臂" if a['side'] == 'R' else "?"
                ts = a['timestamp']
                lines.append(
                    f"  [OK] {a['action']:<10s}  "
                    f"@ {ts:.1f}s (帧 {a['frame']})  "
                    f"{side_label}  {a['angle']:.0f}°"
                )
            else:
                lines.append(f"  [X] {a['action']:<10s}  未检测到")

        lines.append("-" * 50)
        if result['all_found']:
            if result['order_valid']:
                lines.append("  结论: 全部完成，顺序正确 [OK]")
            else:
                lines.append("  结论: 全部完成，但顺序异常 [X]")
        else:
            missing = [a['action'] for a in result['actions'] if not a['found']]
            lines.append(f"  结论: 缺失动作 — {', '.join(missing)}")

        lines.append(f"  共检测到 {result['total_events']} 个事件")
        lines.append("=" * 50)
        return "\n".join(lines)
