"""
Bit Manipulation

Bit manipulation involves operations on individual bits of numbers.
It's not an algorithm or data structure, but a technique for optimization
and low-level operations.

Key operations:
- AND (&): Both bits must be 1 to get 1
- OR (|): At least one bit must be 1 to get 1
- XOR (^): Exactly one bit must be 1 to get 1
- NOT (~): Flips the bit
- Left shift (<<): Multiply by 2
- Right shift (>>): Divide by 2

Use cases:
- Optimization
- Space efficiency
- Low-level operations
- Counting bits, checking properties

Time: O(1) for single operations, O(log n) for counting bits
Space: O(1)
"""


def count_bits(n):
    """
    Count number of 1 bits in binary representation of integer.
    
    Algorithm:
    1. While n > 0:
       a. Check rightmost bit: if n & 1 == 1, increment count
       b. Shift right: n = n >> 1 (same as n // 2)
    2. Return count
    
    Key insight: Use n & 1 to check if rightmost bit is 1.
    Then shift right to process next bit.
    
    Example: count_bits(23)
      23 = 10111 (binary)
      Step 1: 10111 & 1 = 1 → count = 1, shift → 1011
      Step 2: 1011 & 1 = 1 → count = 2, shift → 101
      Step 3: 101 & 1 = 1 → count = 3, shift → 10
      Step 4: 10 & 1 = 0 → count = 3, shift → 1
      Step 5: 1 & 1 = 1 → count = 4, shift → 0
      Result: 4 bits set to 1
    
    Time: O(log n) - number of bits in n
    Space: O(1)
    """
    count = 0
    while n > 0:
        if n & 1 == 1:  # Check if rightmost bit is 1
            count += 1
        n = n >> 1  # Shift right (divide by 2)
    return count


def demo_bit_operations():
    """
    Demonstrate basic bit operations.
    """
    print("=== Bit Operations Demo ===\n")
    
    print("1. AND (&):")
    print("   Truth Table:")
    print("   A | B | A & B")
    print("   --|---|------")
    print("   0 | 0 |   0")
    print("   0 | 1 |   0")
    print("   1 | 0 |   0")
    print("   1 | 1 |   1  ✓")
    print("   Example: 1 & 1 =", 1 & 1)
    print("   Example: 5 & 3 =", 5 & 3, "(101 & 011 = 001)")
    print()
    
    print("2. OR (|):")
    print("   Truth Table:")
    print("   A | B | A | B")
    print("   --|---|------")
    print("   0 | 0 |   0")
    print("   0 | 1 |   1  ✓")
    print("   1 | 0 |   1  ✓")
    print("   1 | 1 |   1  ✓")
    print("   Example: 1 | 0 =", 1 | 0)
    print("   Example: 5 | 3 =", 5 | 3, "(101 | 011 = 111)")
    print()
    
    print("3. XOR (^):")
    print("   Truth Table:")
    print("   A | B | A ^ B")
    print("   --|---|------")
    print("   0 | 0 |   0")
    print("   0 | 1 |   1  ✓")
    print("   1 | 0 |   1  ✓")
    print("   1 | 1 |   0")
    print("   Example: 0 ^ 1 =", 0 ^ 1)
    print("   Example: 5 ^ 3 =", 5 ^ 3, "(101 ^ 011 = 110)")
    print()
    
    print("4. NOT (~):")
    print("   Truth Table:")
    print("   A | ~A")
    print("   --|----")
    print("   0 |  1  (flip)")
    print("   1 |  0  (flip)")
    print("   Example: ~1 =", ~1, "(in Python, uses two's complement)")
    print()
    
    print("5. Left Shift (<<): Multiply by 2")
    print("   001 (1) << 1 =", 1 << 1, "(010 = 2)")
    print("   010 (2) << 1 =", 2 << 1, "(100 = 4)")
    print("   100 (4) << 1 =", 4 << 1, "(1000 = 8)")
    print()
    
    print("6. Right Shift (>>): Divide by 2")
    print("   1000 (8) >> 1 =", 8 >> 1, "(0100 = 4)")
    print("   0100 (4) >> 1 =", 4 >> 1, "(0010 = 2)")
    print("   0010 (2) >> 1 =", 2 >> 1, "(0001 = 1)")
    print()


def demo_count_bits():
    """
    Demonstrate counting 1 bits.
    """
    print("=== Counting Bits Demo ===\n")
    
    n = 23
    print(f"Problem: Count 1 bits in {n}")
    print(f"Binary representation: {bin(n)} = {n:05b}")
    print()
    
    print("Process:")
    print("  Step 1: Check rightmost bit")
    print(f"    {n:05b} & 00001 = {n & 1:05b} → {'1 bit found!' if n & 1 else '0 bit'}")
    count = count_bits(n)
    print(f"  Result: {count} bits set to 1")
    print()
    
    # Show step by step
    print("Step-by-step:")
    temp_n = n
    step = 1
    while temp_n > 0:
        bit = temp_n & 1
        print(f"  Step {step}: {temp_n:05b} & 00001 = {bit:05b} → {'1 bit found!' if bit else '0 bit'}")
        temp_n = temp_n >> 1
        step += 1
    print()
    
    # Test with multiple numbers
    print("Examples:")
    test_cases = [0, 1, 2, 5, 15, 23, 255]
    for num in test_cases:
        bits = count_bits(num)
        print(f"  {num:3d} ({num:08b}) → {bits} bits")
    print()
    
    print("=== Common Bit Tricks ===")
    print("Check if even/odd:")
    print("  n & 1 == 0 → even")
    print("  n & 1 == 1 → odd")
    print()
    print("Multiply by 2: n << 1")
    print("Divide by 2: n >> 1")
    print()
    print("Check if power of 2:")
    print("  n & (n - 1) == 0 → power of 2")
    print("  Example: 8 & 7 = 0 → 8 is power of 2 ✓")
    print("  Example: 7 & 6 = 6 → 7 is not power of 2")


def demo():
    """
    Main demo function for bit manipulation.
    """
    print("=== Bit Manipulation Demo ===\n")
    
    demo_bit_operations()
    print("="*50 + "\n")
    demo_count_bits()


if __name__ == "__main__":
    demo()
