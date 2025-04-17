import regex

def replace_emojis_with_unique_symbols(df):
    """Replaces emojis in DataFrame column names with unique identifiers."""
    emoji_pattern = regex.compile(r'\p{So}|\p{Cn}')

    def replace_emojis(column_name):
        # Find all emojis using regex
        emojis = regex.findall(r'\X', column_name)  # Match grapheme clusters
        replacements = {}
        for emoji in emojis:
            # Check if it's an emoji (Unicode "Other Symbols" or related categories)
            if emoji_pattern.match(emoji):
                # Replace emoji with a unique symbol based on its Unicode representation
                unique_symbol = f"_IMG{'_'.join(f'{ord(c):X}' for c in emoji)}_"
                replacements[emoji] = unique_symbol

        # Replace all emojis in one go
        for emoji, unique_symbol in replacements.items():
            column_name = column_name.replace(emoji, unique_symbol)
        return column_name

    # Apply the replacement function to all column names
    df.columns = df.columns.map(replace_emojis)
    return df 