---
name: fix-todos
description: Analyzes the items in TODO.md and spins up subagents to address them. Use when the user asks to fix, address, or burn down todos
---

Fix the current set of todos:

1. Create a subagent for each item in @TODO.md, and let it analyze and triage what it would take to implement a fix.
2. If there are any open questions about the items or the corresponding fixes, ask the user for clarification.
3. Put the items in a sensible order, such that simple items and items that others depend on get addressed first.
4. Print a table with the sorted list of todos to be addressed.
5. Work through the list of items.

For each todo item, spin up a subagent so you don't pollute your context.
Don't run subagents in parallel, since they will get into each others way when modifying, building, and committing files.
The subagent should:

1. Plan a fix for the todo item
2. Execute the plan
3. Remove the corresponding item from TODO.md
4. Commit the changes

Additional instructions provided by the user: $ARGUMENTS
