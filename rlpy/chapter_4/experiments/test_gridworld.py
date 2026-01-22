from chapter_3.envs.gridworld import Gridworld

def run_basic_tests():
    env = Gridworld()

    U = (-1, 0)
    D = ( 1, 0)
    L = (0, -1)
    R = (0,  1)

    # 1) Off-grid: from top-left, going Up stays put and gives -1
    s = (0, 0)
    s2, r = env.step(s, U)
    assert s2 == (0, 0) and r == -1, (s2, r)

    # 2) Normal move: from (0,0), going Right moves to (0,1) and gives 0
    s2, r = env.step((0, 0), R)
    assert s2 == (0, 1) and r == 0, (s2, r)

    # 3) Special A: from A, ANY action teleports to A' and gives +10
    s2, r = env.step(env.A, U)
    assert s2 == env.A_prime and r == 10, (s2, r)

    # 4) Special B: from B, ANY action teleports to B' and gives +5
    s2, r = env.step(env.B, L)
    assert s2 == env.B_prime and r == 5, (s2, r)

    # A override should win even if action would be off-grid normally
    # (A is at (0,1); Up would be off-grid, but we still go to A')
    s2, r = env.step(env.A, U)
    assert s2 == env.A_prime and r == 10

    print("Basic Gridworld tests passed.")

if __name__ == "__main__":
    run_basic_tests()
