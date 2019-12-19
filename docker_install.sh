#!/bin/bash

INSTALL_DIR=$1

SCRIPTS=($(find "$INSTALL_DIR/src" -maxdepth 1 -type f -name *.py | sort))

for SCRIPT in "${SCRIPTS[@]}"; do
	BASENAME=$(basename $SCRIPT | sed -e "s/.py$//g")
	printf "#!/bin/bash\n\npython3 $SCRIPT \"\$@\"" > "/usr/bin/$BASENAME"
	chmod +x "/usr/bin/$BASENAME"
done

cp $INSTALL_DIR/config/unix/* $HOME
