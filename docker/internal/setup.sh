#!/bin/bash

# setup script to run within the image during build

INSTALL_DIR=$1

# get all python scripts in the src directory
mapfile -t SCRIPTS < <(find "$INSTALL_DIR/src" -maxdepth 1 -type f -name "*.py" | sort)

# create wrapper scripts in /usr/bin which call the individual scripts
for SCRIPT in "${SCRIPTS[@]}"; do
	BASENAME=$(basename "$SCRIPT" | sed -e "s/.py$//g")
	printf "#!/bin/bash\n\npython3 %s \"\$@\"" "$SCRIPT" > "/usr/bin/$BASENAME"
	chmod +x "/usr/bin/$BASENAME"
done

# copy configs from config/unix to /etc
mkdir /etc/convnet_euso
for CONFIGFILE in "$INSTALL_DIR"/config/unix/*.ini; do
  CONFIG_FILENAME=$(basename "$CONFIGFILE")
  # $XDG_RUNTIME_DIR is not set within image, just use /tmp for tensorflow logs
  sed -e "s&logdir=.*&logdir=/tmp&g" "$CONFIGFILE" > "/etc/convnet_euso/$CONFIG_FILENAME"
done

# make internal scripts executable
for SCRIPT in "$INSTALL_DIR"/docker/internal/*.sh; do
  chmod +x "$SCRIPT"
done
