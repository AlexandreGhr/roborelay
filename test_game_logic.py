from env.world import World

print("=== Test 1 : victoire (objet livré avant max_turns) ===")
w = World(5, 5, max_turns=20)
carrier = w.add_carrier("carrier-1", 0, 1)
placer = w.add_placer("placer-1", 4, 2)
w.add_object(1, 1)

print(w.grid)
print("is_done avant de jouer:", w.is_done())
print()

carrier.pick_up("right")   # ramasse l'objet en (1,1)
w.increment_turn()
carrier.move("right")      # (0,1) -> (1,1)
w.increment_turn()
carrier.move("right")      # (1,1) -> (2,1)
w.increment_turn()
carrier.move("right")      # (2,1) -> (3,1), adjacent à la case au-dessus du Placer
w.increment_turn()
carrier.drop("right")      # dépose en (4,1), case vide adjacente au Placer
w.increment_turn()

print("Après livraison du carrier :")
print(w.grid)

placer.charge("up")        # le Placer récupère l'objet posé en (4,1)
w.increment_turn()

print()
print("Score:", w.total_score())
print("is_done après livraison:", w.is_done())
print()

print("=== Test 2 : timeout (rien livré, max_turns atteint) ===")
w2 = World(5, 5, max_turns=3)
w2.add_carrier("carrier-1", 0, 0)
w2.add_placer("placer-1", 4, 4)
w2.add_object(2, 2)

for i in range(3):
    w2.increment_turn()

print("turn_count:", w2.turn_count)
print("Score:", w2.total_score())
print("is_done (timeout):", w2.is_done())