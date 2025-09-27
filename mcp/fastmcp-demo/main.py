import random
from fastmcp import FastMCP


mcp =FastMCP(name="demo-server")

@mcp.tool
def roll_dice(sides: int = 1) -> list[int]:
    """Roll a dice with the given number of sides."""
    return [random.randint(1, sides) for _ in range(sides)]


@mcp.tool
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


if __name__ == "__main__":
    mcp.run()