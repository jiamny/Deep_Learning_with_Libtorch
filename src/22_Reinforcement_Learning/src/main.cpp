
#include "Trainer.h"
#include <iostream>
#include <unistd.h>

int main() {

	std::cout << "Current path is " << get_current_dir_name() << '\n';

    Trainer trainer(3, 18, 100000);
    trainer.train(123, "src/22_Reinforcement_Learning/atari_roms/pong.bin", 100000);

}
