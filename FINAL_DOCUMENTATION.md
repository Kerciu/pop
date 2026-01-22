# Raport Końcowy Projektu: Optymalizacja Logistyki Saniami Świętego Mikołaja (Santa Tracker - Hash Code 2022)

**Autorzy:** Kacper Górski i Aleksander Paliwoda

---

## 1. Wstęp i Definicja Problemu

Celem projektu było opracowanie systemu sterowania dla sań Świętego Mikołaja w środowisku symulacyjnym, odwzorowującym zadanie finałowe Google Hash Code 2022. Problem ten charakteryzuje się wysoką złożonością, łącząc trzy odrębne domeny optymalizacji:

1.  **Problem Komiwojażera (TSP):** Konieczność wyznaczenia optymalnej kolejności odwiedzania celów na mapie 2D.
2.  **Problem Plecakowy (Knapsack Problem):** Decyzja o doborze prezentów w taki sposób, aby zmaksymalizować wynik punktowy przy ograniczonym udźwigu sań, uwzględniając jednocześnie wagę paliwa (marchewek).
3.  **Sterowanie Fizyczne (Control Theory):** Poruszanie się w środowisku ciągłym bez tarcia, gdzie każde przyspieszenie zużywa zasoby, a bezwładność utrudnia precyzyjne manewry.



### 1.1. Środowisko Symulacyjne i Fizyka
Kluczowym wyzwaniem była fizyka ruchu w środowisku bez tarcia. Pozycja sań $P$ w czasie $t+1$ zależy od prędkości $V$, która pozostaje stała, dopóki nie zostanie wykonana akcja przyspieszenia:

$$P_{t+1} = P_t + V_t$$

Dodatkowym ograniczeniem jest masa całkowita (sanie + prezenty + paliwo), która determinuje maksymalne możliwe przyspieszenie. Przeładowanie sań uniemożliwia sterowanie.

---

## 2. Dane Treningowe i Inżynieria Scenariuszy

Jednym z kluczowych wniosków projektu było odkrycie, że jakość i struktura danych wejściowych mają krytyczny wpływ na zdolność modelu RL do nauki. Standardowe pliki z zadania Hash Code okazały się nieodpowiednie do początkowej fazy treningu.

### 2.1. Problemy z Dostępnymi Danymi
Podczas eksperymentów zidentyfikowano dwa główne problemy:
1.  **Małe zbiory danych:** Przy niewielkiej liczbie prezentów rozsianych losowo po mapie, prawdopodobieństwo, że agent (poruszający się losowo w fazie eksploracji) trafi w prezent, było bliskie zeru. Brak wczesnych nagród uniemożliwiał zbieżność sieci.
2.  **Plik `b_better_hurry.in.txt`:** Analiza tego oficjalnego zbioru wykazała, że zawiera on 4 klastry prezentów oddalone bardzo daleko od bazy. Agent DQN, nie posiadając jeszcze wyuczonej strategii oszczędzania paliwa, nie był w stanie dolecieć do żadnego z klastrów przed wyczerpaniem zasobów lub czasu, co skutkowało brakiem pozytywnego sygnału zwrotnego.

### 2.2. Rozwiązanie: Własny Generator Danych (`generate_input.py`)
Aby umożliwić agentowi naukę (Curriculum Learning), stworzono algorytm generujący mapę `huge_challenge.in.txt`. Jego celem było stworzenie gęstych klastrów prezentów w relatywnie bliskiej odległości od bazy, co zwiększało szansę na "przypadkowy sukces" w początkowej fazie treningu.

---

## 3. Rozwiązania Algorytmiczne (Baseline)

Przed przystąpieniem do trenowania modeli AI, zaimplementowano rozwiązania klasyczne, służące jako punkt odniesienia (benchmark) oraz weryfikacja poprawności symulatora.

### 3.1. Algorytm Zachłanny (Greedy Solver)
Pierwsza iteracja opierała się na prostej heurystyce zaimplementowanej w `greedy_solver.py`: "leć do najbliższego celu".
* **Strategia:** Jeśli sanie mają prezenty, leć do najbliższego celu dostawy. Jeśli są puste, ładuj prezenty. Jeśli brak paliwa, tankuj.
* **Ocena:** Mimo że algorytm ten skutecznie dostarczał prezenty, był dalece nieoptymalny. Brak planowania długoterminowego powodował chaotyczne trasy i częste, nieefektywne powroty do bazy.

### 3.2. Smart Solver (Prototyp Logiczny)
W celu usprawnienia sterowania stworzono rozwiązanie modułowe (`SmartSolver`), oparte na sztywnych regułach ("hardcoded logic"). Był to algorytm "szybkiego reagowania", a nie docelowe rozwiązanie produkcyjne. Wykorzystywał on:

* **Analityczny Model Hamowania:** W module `motion_control.py` zaimplementowano funkcję `get_stopping_distance(v)`, która wylicza drogę potrzebną do zatrzymania według wzoru:
    $$d = \frac{|v| \cdot (|v| + 1)}{2}$$
    Pozwoliło to na precyzyjne zatrzymywanie się nad celem bez konieczności "uczenia się" fizyki.
* **Planowanie (TSP + Knapsack):** Moduł `route_planner.py` dobierał prezenty do bezpiecznego limitu wagi (`MAX_SAFE_WEIGHT`) i sortował je metodą Najbliższego Sąsiada.

---

## 4. Eksperymenty z Uczeniem Maszynowym (AI)

Główny wysiłek badawczy projektu skoncentrowany był na wytrenowaniu agenta, który samodzielnie nauczyłby się strategii nawigacji i zarządzania zasobami.



### 4.1. Neuroewolucja (Algorytm Genetyczny)
Jako alternatywę dla uczenia ze wzmocnieniem (RL), przeprowadzono eksperyment z podejściem hybrydowym zaimplementowanym w `genetic_agent.py`:
* **Koncept Hybrydowy:** Sztywna logika (w `train_genetic.py`) odpowiadała za decyzje binarne w bazie (Załaduj/Zatankuj), odciążając sieć neuronową, która sterowała wyłącznie wektorem ruchu w terenie.
* **Metoda:** Ewolucja wag sieci neuronowej poprzez dodawanie szumu Gaussa do parametrów (`param.data += noise`) w populacji 50 agentów.
* **Wynik:** Przeprowadzono jedną próbę treningową. Wyniki były przeciętne – agent miał trudności z precyzją sterowania (brak "czucia" momentu hamowania) i nie dorównał skutecznością rozwiązaniom algorytmicznym.

### 4.2. Głębokie Uczenie ze Wzmocnieniem (DQN)
Najwięcej czasu poświęcono architekturze **Dueling DQN** z buforem doświadczeń (Replay Buffer).

#### Architektura
* **Sieć:** Dueling DQN rozdzielająca estymację wartości stanu $V(s)$ od przewagi akcji $A(a)$.
* **Wejście:** Tensor 14-elementowy w `SleighEnv`, zawierający znormalizowane pozycje, prędkości, stan paliwa i flagi logiczne (np. `brake_warning`, `in_range`).

#### Analiza Problemu: "Sukces w Treningu, Porażka w Ewaluacji"
Podczas eksperymentów zaobserwowano interesujące zjawisko:
1.  **Model "Najlepszy" (The Fuel Bug):** Wytrenowano model, który osiągnął niemal perfekcję w nawigacji i dostarczaniu prezentów. Analiza wykazała jednak, że **środowisko treningowe zawierało błąd – brak kosztu paliwa za przyspieszenie**. Agent nauczył się doskonałej fizyki ruchu, ignorując całkowicie ekonomię zasobów.
2.  **Wprowadzenie Kosztu Paliwa:** Po naprawieniu błędu w `sleigh_env.py` (odjęcie paliwa za każdą akcję ruchu), proces uczenia załamał się.
3.  **Porażka w Ewaluacji:** Mimo że w treningu agent podejmował próby dostaw (dzięki losowości eksploracji), w deterministycznej ewaluacji nie dostarczył żadnego prezentu.

**Diagnoza:** Głównym powodem był **Problem Propagacji Sygnału (Credit Assignment)**. Dostarczenie prezentu następuje po sekwencji kilkuset kroków. W gęstej sieci neuronowej informacja o rzadkiej nagrodzie końcowej nie była w stanie skutecznie "przepropagować" się wstecz, aby skorelować zużycie paliwa na starcie z sukcesem na mecie.

---

## 5. Wyzwania Techniczne i Implementacja

### 5.1. Inżynieria Nagród (Reward Shaping)
Funkcja nagrody w `sleigh_env.py` przeszła znaczącą ewolucję:
* **Nagroda Główna:** +1000 pkt za `Deliver` (zwiększona, by przebić szum).
* **Bezpieczniki (Safety):** Wprowadzenie kar (-1.0 do -2.0) za próby wykonania niemożliwych akcji (np. `Load` poza bazą) oraz dużej kary (-200.0) za wyczerpanie paliwa.
* **Shaping:** Zredukowana nagroda za zbliżanie się do celu (`diff * 0.1`), aby uniknąć pętli lokalnych maximów.

### 5.2. Normalizacja i Generowanie Danych
* **Dynamiczne Skalowanie:** Początkowo sieć nie uczyła się, traktując małe wartości współrzędnych jako szum. Wprowadzono dynamiczne skalowanie w `SleighEnv` (`max_coord`), normalizujące wejście do zakresu $[-1, 1]$.
* **Generator Danych:** Stworzono własny skrypt generujący, który tworzył klastry prezentów, symulując bardziej realistyczne i łatwiejsze do nauczenia scenariusze niż w pełni losowy szum.

---

## 6. Podsumowanie i Wnioski Końcowe

Projekt Hash Code 2022 stanowił studium przypadku ograniczeń metod Deep Reinforcement Learning w deterministycznych środowiskach fizycznych.

1.  **Wyższość Analityki nad Aproksymacją:** W środowisku o znanych prawach fizyki, proste algorytmy wykorzystujące wzory matematyczne (jak w `motion_control.py`) są znacznie pewniejsze niż sieci neuronowe, które muszą aproksymować te prawa metodą prób i błędów.
2.  **Złożoność Problemu:** Połączenie problemu plecakowego, TSP i fizyki okazało się zbyt złożone dla standardowego agenta DQN bez implementacji hierarchicznego uczenia (Hierarchical RL).
3.  **Wnioski z Porażki DQN:** Fakt, że model działał idealnie *bez* kosztów paliwa, a przestał działać *z* kosztami, wskazuje, że wąskim gardłem nie była nawigacja, lecz planowanie ekonomiczne na długim horyzoncie czasowym.