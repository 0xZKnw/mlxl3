#!/bin/bash
# Install the latest official Desktop and its bundled CLI, without sudo or Node.
set -euo pipefail

work_dir='' stage_dir='' app_dir='' backup='' profile_temp='' lock_dir=''
cli_path='' cli_target='' cli_backup_dir='' log_file=''
mounted=0 swapped=0 committed=0 linked=0 locked=0
reported=0 cancelled=0 phase='starting' recovery_hint='Run the command again. If it still fails, share the log shown below.'
installer_argc=$#
bold='' dim='' accent='' good='' bad='' reset=''
if [[ -t 1 && -z ${NO_COLOR+x} && ${TERM:-dumb} != dumb ]]; then
  bold=$'\033[1m' dim=$'\033[2m' accent=$'\033[36m' good=$'\033[32m' bad=$'\033[31m' reset=$'\033[0m'
fi

banner() { printf '\n%sMLXL3%s  %sDesktop + CLI installer%s\n' "$bold" "$reset" "$dim" "$reset"; }
step() {
  phase=$2
  printf '\n%s[%s/5]%s %s%s%s\n' "$accent" "$1" "$reset" "$bold" "$phase" "$reset"
  if [[ -n $log_file ]]; then printf '[%s/5] %s\n' "$1" "$phase" >>"$log_file"; fi
}
note() {
  printf '  %s%s%s\n' "$dim" "$*" "$reset"
  if [[ -n $log_file ]]; then printf '%s\n' "$*" >>"$log_file"; fi
}
show_error() {
  local heading='Installation stopped'
  local style="$bad$bold" ending=$reset
  if [[ ! -t 2 ]]; then style='' ending=''; fi
  if [[ $committed == 1 ]]; then heading='Installed; cleanup needs attention'; fi
  reported=1
  printf '\n%s! MLXL3: %s%s\n  Step: %s\n  Cause: %s\n\n  Next: %s\n' "$style" "$heading" "$ending" "$phase" "$1" "${2:-$recovery_hint}" >&2
  if [[ -n $log_file ]]; then printf '  Details: %s\n' "$log_file" >&2; fi
  if [[ -n $log_file ]]; then printf 'Step: %s\nCause: %s\nNext: %s\n' "$phase" "$1" "${2:-$recovery_hint}" >>"$log_file"; fi
}
fail() {
  show_error "$@"
  exit 1
}

cleanup() {
  local status=$?
  set +e
  trap - EXIT HUP INT TERM
  if [[ $committed == 0 && $swapped == 1 ]]; then
    if ! mv "$app_dir" "$stage_dir/failed.app"; then
      printf 'Could not undo installation. Previous app: %s\n' "$backup" >&2
      stage_dir=''
      status=1
    fi
  fi
  if [[ $committed == 0 && -n $backup && -d $backup ]]; then
    if [[ -e $app_dir || -L $app_dir ]] || ! mv "$backup" "$app_dir"; then
      printf 'Restore the previous app from: %s\nRestore it to: %s\n' "$backup" "$app_dir" >&2
      stage_dir=''
      status=1
    else
      note 'Your previous Desktop app was restored.'
    fi
  fi
  if [[ $committed == 0 && $linked == 1 && -L $cli_path && $(readlink "$cli_path") == "$cli_target" ]] && ! rm -f "$cli_path" 2>>"$log_file"; then
    show_error 'The new CLI link could not be removed during rollback.' 'Use the previous CLI backup shown below to restore your command manually.'
    status=1
  fi
  if [[ $committed == 0 && -n $cli_backup_dir ]]; then
    if [[ -e $cli_backup_dir/mlxl3 || -L $cli_backup_dir/mlxl3 ]]; then
      if [[ -e $cli_path || -L $cli_path ]] || ! mv "$cli_backup_dir/mlxl3" "$cli_path"; then
        printf 'Previous CLI kept at: %s/mlxl3\nRestore it to: %s\n' "$cli_backup_dir" "$cli_path" >&2
        cli_backup_dir=''
        status=1
      else
        note 'Your previous CLI command was restored.'
      fi
    fi
    if [[ -n $cli_backup_dir ]] && ! rmdir "$cli_backup_dir" 2>>"$log_file"; then
      show_error 'An empty CLI backup folder could not be removed.' "Check $cli_backup_dir in Finder. Your previous command has already been restored."
      status=1
    fi
  fi
  if [[ $mounted == 1 ]]; then
    if ! hdiutil detach "$work_dir/mount" >>"$log_file" 2>&1 \
      && ! hdiutil detach -force "$work_dir/mount" >>"$log_file" 2>&1; then
      printf 'Could not unmount installer: %s/mount\n' "$work_dir" >&2
      show_error 'macOS could not eject the installer disk.' 'Eject the installer disk in Finder. Your app and CLI installation status is shown above.'
      work_dir=''
      status=1
    fi
  fi
  if [[ -n $profile_temp ]] && ! rm -f "$profile_temp" 2>>"$log_file"; then
    show_error 'A temporary shell configuration file could not be removed.' "Remove $profile_temp in Finder. The original shell profile was preserved."
    status=1
  fi
  if [[ -n $stage_dir ]] && ! rm -rf "$stage_dir" 2>>"$log_file"; then
    show_error 'The temporary app folder could not be removed.' "Your app and CLI status is shown above. Remove $stage_dir in Finder when you no longer need its previous-app backup."
    status=1
  fi
  if [[ -n $work_dir ]] && ! rm -rf "$work_dir" 2>>"$log_file"; then
    show_error 'The downloaded installer files could not be removed.' "Remove $work_dir in Finder after ejecting any installer disk."
    status=1
  fi
  if [[ $locked == 1 ]] && ! rmdir "$lock_dir" 2>>"$log_file"; then
    show_error 'The installer lock folder could not be released.' "Once no installer is running, remove $lock_dir before trying another update."
    status=1
  fi
  if [[ $status != 0 && $reported == 0 ]]; then
    if [[ $cancelled == 1 ]]; then
      show_error 'The command was interrupted.' 'Run the same command again when you are ready.'
    else show_error 'A system command could not complete this step. See Details for the technical output.'; fi
  fi
  exit "$status"
}

field() { plutil -extract "releases.$1" raw -expect "${2:-string}" -o - "$work_dir/releases.json" 2>>"$log_file"; }

main() {
  [[ $# == "$installer_argc" ]] || fail 'Incomplete installer. Download it again.' 'Run the curl command again; the script download ended before its arguments were read.'
  local channel=stable version major minor page count index tag date draft prerelease selected='' latest=''
  local asset_index name size digest url selected_name='' selected_url='' selected_digest='' selected_size=''
  local expected_prefix build bundle_version candidate_version current_version='' current_build='' runtime profile path_line parent system_app
  export LC_ALL=C
  if [[ ${1:-} == --help && $# == 1 ]]; then
    banner
    printf '\nInstall or update the latest Desktop app and its bundled CLI.\n\nUsage: bash install.sh [--channel stable|beta]\n\n  --channel stable   Latest stable release (default)\n  --channel beta     Latest beta release\n\nExisting CLI commands are backed up before replacement.\nModels and conversations are kept. Requires Apple Silicon and macOS 26.2+.\n'
    return
  fi
  if [[ $# != 0 ]]; then
    [[ $# == 2 && $1 == --channel ]] || fail 'These installer options are not supported.' 'Use --channel stable or --channel beta. Run --help for examples.'
    channel=$2
  fi
  [[ $channel == stable || $channel == beta ]] || fail 'This release channel does not exist.' 'Choose --channel stable or --channel beta.'
  banner
  note "Channel: $channel"
  step 1 'Check this Mac'
  recovery_hint='Use an Apple Silicon Mac running macOS 26.2 or newer. If your terminal uses Rosetta, reopen it in native mode.'
  [[ $(uname -s) == Darwin && $(uname -m) == arm64 ]] || fail 'This installer needs an Apple Silicon Mac and a native arm64 terminal.'
  version=$(sw_vers -productVersion) || fail 'The macOS version could not be read.'
  [[ $version =~ ^([0-9]{1,3})\.([0-9]{1,3})(\.[0-9]{1,3})?$ ]] || fail 'Cannot read the macOS version.'
  major=$((10#${BASH_REMATCH[1]})) minor=$((10#${BASH_REMATCH[2]}))
  ((major > 26 || (major == 26 && minor >= 2))) || fail 'Requires macOS 26.2 or later.'
  [[ ${HOME:-} == /* && $HOME != / ]] || fail 'HOME must be an absolute user directory.'
  note "Apple Silicon · macOS $version"
  mkdir -p "$HOME/Library/Logs/MLXL3" || fail 'Your home folder is not writable.' 'Check your home folder permissions, then try again.'
  log_file=$(mktemp "$HOME/Library/Logs/MLXL3/install.XXXXXX") || fail 'An installer log could not be created.' 'Check free disk space and the permissions of ~/Library/Logs/MLXL3, then retry.'
  printf 'MLXL3 installer · channel: %s\n' "$channel" >"$log_file"
  cli_path="$HOME/.local/bin/mlxl3"
  lock_dir="$HOME/Library/Caches/io.mlxl3.install.lock"
  mkdir -p "${lock_dir%/*}" || fail 'The installer cache folder is not writable.' 'Check your ~/Library/Caches permissions, then retry.'
  mkdir "$lock_dir" 2>/dev/null || fail 'Another installer is running, or a previous run was interrupted.' "Let the other installer finish. If none is running, remove the empty lock folder at $lock_dir and retry."
  locked=1
  trap cleanup EXIT
  trap 'cancelled=1; exit 130' HUP INT TERM
  work_dir=$(mktemp -d "${TMPDIR:-/tmp}/mlxl3-install.XXXXXX") || fail 'A temporary installation folder could not be created.' 'Check free disk space and your temporary-folder permissions, then retry.'
  step 2 'Find the latest release'
  recovery_hint='Try the other channel, or report this release and the log at https://github.com/0xZKnw/mlxl3/issues.'
  for ((page = 1; page <= 10; page++)); do
    curl --fail --silent --show-error --location --proto '=https' --proto-redir '=https' \
      --connect-timeout 15 --max-time 60 --max-filesize 16777216 \
      "https://api.github.com/repos/0xZKnw/mlxl3/releases?per_page=100&page=$page" -o "$work_dir/release-page.json" 2>>"$log_file" \
      || fail 'Could not get the release list from GitHub.' 'Check your internet connection and retry. GitHub may be temporarily unavailable or limiting requests.'
    {
      printf '{"releases":'
      cat "$work_dir/release-page.json"
      printf '}'
    } >"$work_dir/releases.json"
    count=$(plutil -extract releases raw -expect array -o - "$work_dir/releases.json" 2>>"$log_file") || fail 'GitHub returned release information that could not be read.'
    [[ $count =~ ^[0-9]{1,3}$ ]] || fail 'Invalid release page size.'
    ((10#$count <= 100)) || fail 'Invalid release page size.'
    for ((index = 0; index < count; index++)); do
      tag=$(field "$index.tag_name") || fail 'Invalid release tag.'
      [[ $tag =~ ^v[0-9]+\.[0-9]+\.[0-9]+(-[A-Za-z0-9][A-Za-z0-9.-]*)?$ ]] || continue
      draft=$(field "$index.draft" bool) || fail 'Invalid release draft status.'
      [[ $draft == false ]] || continue
      prerelease=$(field "$index.prerelease" bool) || fail 'Invalid release channel.'
      if [[ $channel == stable ]]; then
        [[ $prerelease == false ]] || continue
      else [[ $prerelease == true ]] || continue; fi
      date=$(field "$index.published_at")
      [[ $date =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z$ ]] || fail 'Invalid release publication date.'
      if [[ -z $selected || $date > $latest ]]; then
        selected=$index
        latest=$date
        cp "$work_dir/releases.json" "$work_dir/selected.json"
      fi
    done
    ((count == 100)) || break
  done
  ((page <= 10)) || fail 'More than 1,000 releases: cannot guarantee the latest version. Update the installer.'
  [[ -n $selected ]] || fail "No published Desktop release found for $channel."
  mv "$work_dir/selected.json" "$work_dir/releases.json"
  tag=$(field "$selected.tag_name")
  expected_prefix="MLXL3-Desktop-${tag}-b"
  count=$(field "$selected.assets" array) || fail 'Invalid release assets.'
  [[ $count =~ ^[0-9]{1,3}$ ]] || fail 'Too many release assets to validate safely.'
  ((10#$count <= 100)) || fail 'Too many release assets to validate safely.'
  for ((asset_index = 0; asset_index < count; asset_index++)); do
    name=$(field "$selected.assets.$asset_index.name") || fail 'Invalid release asset name.'
    [[ $name == "$expected_prefix"*'-Apple-Silicon.dmg' ]] || continue
    build=${name#"$expected_prefix"}
    build=${build%-Apple-Silicon.dmg}
    [[ $build =~ ^[0-9]{1,9}$ ]] || fail 'Invalid Desktop build number.'
    [[ -z $selected_name ]] || fail 'Ambiguous Desktop release: multiple disk images.'
    url=$(field "$selected.assets.$asset_index.browser_download_url")
    digest=$(field "$selected.assets.$asset_index.digest")
    size=$(field "$selected.assets.$asset_index.size" integer)
    [[ $url == "https://github.com/0xZKnw/mlxl3/releases/download/$tag/$name" ]] || fail 'Unexpected download URL.'
    [[ $digest =~ ^sha256:[0-9a-f]{64}$ ]] || fail 'The release has no valid SHA-256 digest.'
    [[ $size =~ ^[0-9]{1,9}$ ]] || fail 'Invalid download size.'
    ((10#$size > 0 && 10#$size <= 536870912)) || fail 'Invalid download size.'
    selected_name=$name selected_url=$url selected_digest=${digest#sha256:} selected_size=$size
  done
  [[ -n $selected_name ]] || fail 'The latest release has no Apple Silicon Desktop disk image.'
  build=${selected_name#"$expected_prefix"}
  build=${build%-Apple-Silicon.dmg}
  version=${tag#v}
  bundle_version=${version%%-*}
  note "Selected: Desktop $version · build $build"
  app_dir="$HOME/Applications/MLXL3 Desktop.app"
  if [[ ! -e $app_dir && ! -L $app_dir ]]; then
    system_app=$(find /Applications -maxdepth 1 -type d -name 'MLXL3 Desktop.app' -print -quit 2>/dev/null || true)
    [[ -z $system_app ]] || app_dir=$system_app
  fi
  [[ ! -L $app_dir ]] || fail 'The app in Applications is a symbolic link.' 'Place the actual MLXL3 Desktop app in ~/Applications, then run the command again.'
  runtime="$app_dir/Contents/Resources/runtime/mlxl3"
  cli_target=$runtime
  if [[ -e $app_dir ]]; then
    [[ $(plutil -extract CFBundleIdentifier raw -o - "$app_dir/Contents/Info.plist" 2>>"$log_file") == io.mlxl3.desktop ]] || fail 'The existing item named MLXL3 Desktop is not a recognized MLXL3 app.' "Move $app_dir out of the way before retrying. It was not changed."
    current_version=$(plutil -extract CFBundleShortVersionString raw -o - "$app_dir/Contents/Info.plist")
    current_build=$(plutil -extract CFBundleVersion raw -o - "$app_dir/Contents/Info.plist")
  fi
  note "Installed: ${current_version:-not installed}"
  note "App: $app_dir"
  if [[ -e $cli_path || -L $cli_path ]]; then
    [[ ! -d $cli_path && (-f $cli_path || -L $cli_path) ]] || fail 'The CLI path is a folder or another non-command item.' "Move $cli_path out of the way, then rerun the installer. Nothing at that path was changed."
    if [[ ! -L $cli_path || $(readlink "$cli_path") != "$runtime" ]]; then
      note 'Existing CLI detected. It will be backed up and updated.'
      if [[ -L $cli_path ]]; then note "Current CLI link: $(readlink "$cli_path")"; fi
    fi
  fi
  profile="$HOME/.zprofile"
  recovery_hint='Check that your home folder and shell configuration are writable, then retry.'
  local login_shell=${SHELL:-/bin/zsh}
  [[ ${login_shell##*/} != bash ]] || profile="$HOME/.bash_profile"
  # Expanded by the next login shell, rather than hardcoding this process's PATH.
  # shellcheck disable=SC2016
  path_line='export PATH="$HOME/.local/bin:$PATH"'
  case "$PATH" in
  "$HOME/.local/bin" | "$HOME/.local/bin:"*) ;;
  *)
    [[ ! -L $profile && (! -e $profile || -f $profile) ]] || fail 'Your shell profile is a link or another non-regular file, so it was not edited.' 'Add ~/.local/bin to PATH in your shell configuration, open a new terminal, then retry.'
    if ! grep -Fqx "$path_line" "$profile" 2>/dev/null; then
      profile_temp=$(mktemp "$HOME/.mlxl3-profile.XXXXXX")
      [[ ! -e $profile ]] || cp -p "$profile" "$profile_temp"
      printf '\n# MLXL3 command line\n%s\n' "$path_line" >>"$profile_temp"
    fi
    ;;
  esac
  if [[ ($current_version != "$version" && $current_version != "$bundle_version") || $current_build != "$build" ]]; then
    if pgrep -x MLXL3Studio >/dev/null; then fail 'MLXL3 Desktop is still running.' 'Quit the app, then rerun the same command. Your app and CLI have not been changed.'; fi
    parent=${app_dir%/*}
    mkdir -p "$parent" "$HOME/.local/bin"
    [[ -w $parent ]] || fail "Cannot write to $parent. Move the app to ~/Applications and retry."
    stage_dir=$(mktemp -d "$parent/.mlxl3-install.XXXXXX")
    step 3 'Download and verify'
    recovery_hint='Check your internet connection and free disk space, then retry. The previous app is kept until the download passes verification.'
    note "Downloading Desktop $version from GitHub · $(((10#$selected_size + 999999) / 1000000)) MB"
    curl --fail --silent --show-error --location --proto '=https' --proto-redir '=https' --connect-timeout 15 \
      --max-time 300 --max-filesize "$selected_size" "$selected_url" -o "$work_dir/desktop.dmg" 2>>"$log_file" \
      || fail 'The Desktop download could not finish.'
    [[ $(wc -c <"$work_dir/desktop.dmg" | tr -d '[:space:]') == "$selected_size" ]] || fail 'The downloaded file is incomplete.'
    printf '%s  %s\n' "$selected_digest" "$work_dir/desktop.dmg" | shasum -a 256 -c - >>"$log_file" 2>&1 \
      || fail 'The downloaded file failed its SHA-256 integrity check.' 'Run the command again to download a fresh copy. The rejected file will not be installed.'
    note 'Download verified.'
    if [[ -n $current_version ]]; then step 4 'Update Desktop'; else step 4 'Install Desktop'; fi
    recovery_hint='Close MLXL3 Desktop and check that your Applications folder is writable, then retry.'
    mkdir "$work_dir/mount"
    mounted=1
    hdiutil attach -readonly -nobrowse -mountpoint "$work_dir/mount" "$work_dir/desktop.dmg" >>"$log_file" 2>&1 \
      || fail 'macOS could not open the downloaded disk image.' 'Retry the download. If it still fails, share the log shown in Details.'
    ditto "$work_dir/mount/MLXL3 Desktop.app" "$stage_dir/new.app" >>"$log_file" 2>&1 \
      || fail 'The app could not be copied into Applications.' 'Check that there is enough free disk space and that the Applications folder is writable.'
    [[ $(plutil -extract CFBundleIdentifier raw -o - "$stage_dir/new.app/Contents/Info.plist" 2>>"$log_file") == io.mlxl3.desktop ]] || fail 'The disk image contains a different app instead of MLXL3 Desktop.' 'Try the other channel, or report this release and its log.'
    candidate_version=$(plutil -extract CFBundleShortVersionString raw -o - "$stage_dir/new.app/Contents/Info.plist")
    [[ ($candidate_version == "$version" || $candidate_version == "$bundle_version") &&
      $(plutil -extract CFBundleVersion raw -o - "$stage_dir/new.app/Contents/Info.plist") == "$build" ]] || fail 'The app version inside the download does not match the selected release.' 'Try the other channel, or report this release and its log.'
    [[ -x $stage_dir/new.app/Contents/MacOS/MLXL3Studio && -x $stage_dir/new.app/Contents/Resources/runtime/mlxl3 ]] || fail 'The release is missing the Desktop executable or its command-line tool.' 'Try the other channel, or report this incomplete release and its log.'
    codesign --verify --deep --strict "$stage_dir/new.app" >>"$log_file" 2>&1 \
      || fail 'The downloaded app failed its signature check.' 'Try a fresh download. If it still fails, report this release and its log; the previous app is kept.'
    if pgrep -x MLXL3Studio >/dev/null; then fail 'MLXL3 Desktop was opened while the update was downloading.' 'Quit the app and rerun the command. The running app was not replaced.'; fi
    if [[ -e $app_dir ]]; then
      backup="$stage_dir/previous.app"
      mv "$app_dir" "$backup" 2>>"$log_file" || fail 'The current app could not be moved into its temporary backup.'
    fi
    mv "$stage_dir/new.app" "$app_dir" 2>>"$log_file" || fail 'The new app could not be placed in Applications.'
    swapped=1
  else
    step 3 'No download needed'
    note "Desktop $version (build $build) is already current."
    step 4 'Verify installed Desktop'
    [[ -x $runtime ]] || fail 'The installed CLI is missing.'
    codesign --verify --deep --strict "$app_dir" >>"$log_file" 2>&1 \
      || fail 'The installed app failed its signature check.' 'Move the damaged app out of Applications and run the installer again. Keep your models and conversations in place.'
  fi
  step 5 'Set up the command line'
  recovery_hint='Check that ~/.local/bin and your shell profile are writable, then retry. See Details for the system error.'
  mkdir -p "$HOME/.local/bin"
  if [[ -e $cli_path || -L $cli_path ]]; then
    if [[ ! -L $cli_path || $(readlink "$cli_path") != "$runtime" ]]; then
      [[ ! -d $cli_path && (-f $cli_path || -L $cli_path) ]] || fail 'The CLI path changed while downloading and is no longer a command file.' 'Move the item out of ~/.local/bin/mlxl3, then retry.'
      cli_backup_dir=$(mktemp -d "$HOME/.local/bin/.mlxl3-cli-backup.XXXXXX")
      mv "$cli_path" "$cli_backup_dir/mlxl3" 2>>"$log_file" \
        || fail 'The previous CLI could not be saved. It was not replaced.'
    fi
  fi
  if [[ ! -L $cli_path ]]; then
    linked=1
    ln -s "$runtime" "$cli_path" 2>>"$log_file" || fail 'The new CLI link could not be created.'
  fi
  [[ -L $cli_path && $(readlink "$cli_path") == "$runtime" ]] || fail 'The CLI command changed during setup.' 'Check ~/.local/bin/mlxl3 and retry. Its previous version is kept in the backup shown below.'
  "$cli_path" --help >>"$log_file" 2>&1 || fail 'The newly installed CLI could not start.' 'Retry once. If it still fails, share the log; the previous app and command will be restored when possible.'
  note "CLI linked to Desktop $version."
  [[ -z $profile_temp ]] || mv "$profile_temp" "$profile" 2>>"$log_file" || fail 'Your shell profile could not be updated.'
  profile_temp=''
  committed=1
  printf '\n%s✓ MLXL3 is ready%s\n' "$good$bold" "$reset"
  note "Desktop: $version ($channel) · CLI included"
  note "CLI: $cli_path"
  if [[ -n $cli_backup_dir ]]; then note "CLI backup: $cli_backup_dir/mlxl3"; fi
  printf '\n%sNext steps%s\n  1. Open MLXL3 Desktop from Applications.\n  2. Open a new terminal, then run: mlxl3 --help\n' "$bold" "$reset"
  note 'Models and conversations are kept.'
  note 'If macOS blocks the first launch: System Settings → Privacy & Security → Open Anyway.'
}

main "$@"
