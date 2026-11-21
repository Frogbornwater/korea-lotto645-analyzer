#include <array>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <vector>

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "usage: v84_exact_evaluator INPUT OUTPUT\n";
        return 2;
    }
    std::ifstream input(argv[1]);
    std::ofstream output(argv[2]);
    if (!input || !output) return 3;

    int portfolio_count = 0;
    input >> portfolio_count;
    if (portfolio_count <= 0 || portfolio_count > 100) return 4;

    std::vector<std::array<std::uint64_t, 10>> portfolios(portfolio_count);
    for (int p = 0; p < portfolio_count; ++p) {
        for (int t = 0; t < 10; ++t) {
            if (!(input >> portfolios[p][t])) return 5;
        }
    }

    constexpr long long total_outcomes = 8145060LL;
    output << "portfolio_id,any3,any4,any5,any6,total_outcomes\n";
    for (int p = 0; p < portfolio_count; ++p) {
        long long any3 = 0, any4 = 0, any5 = 0, any6 = 0;
        const auto& tickets = portfolios[p];
        for (int a = 0; a < 40; ++a)
        for (int b = a + 1; b < 41; ++b)
        for (int c = b + 1; c < 42; ++c)
        for (int d = c + 1; d < 43; ++d)
        for (int e = d + 1; e < 44; ++e)
        for (int f = e + 1; f < 45; ++f) {
            const std::uint64_t draw = (1ULL << a) | (1ULL << b) | (1ULL << c)
                | (1ULL << d) | (1ULL << e) | (1ULL << f);
            bool hit3 = false, hit4 = false, hit5 = false, hit6 = false;
            for (const std::uint64_t ticket : tickets) {
                const int hits = __builtin_popcountll(draw & ticket);
                if (hits >= 3) hit3 = true;
                if (hits >= 4) hit4 = true;
                if (hits >= 5) hit5 = true;
                if (hits == 6) hit6 = true;
                if (hit3 && hit4 && hit5 && hit6) break;
            }
            any3 += hit3;
            any4 += hit4;
            any5 += hit5;
            any6 += hit6;
        }
        output << p << ',' << any3 << ',' << any4 << ',' << any5 << ','
               << any6 << ',' << total_outcomes << '\n';
        std::cerr << "exactly evaluated portfolio " << (p + 1)
                  << '/' << portfolio_count << '\n';
    }
    return 0;
}
