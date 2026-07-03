const fs = require('fs');
const path = require('path');

const inputDirectory = __dirname;

function findInputFiles(directory) {
	return fs
		.readdirSync(directory)
		.filter((fileName) => /^datatran\d{4}_processado\.csv$/i.test(fileName))
		.sort();
}

function splitCsvLine(line) {
	const values = [];
	let currentValue = '';
	let insideQuotes = false;

	for (let i = 0; i < line.length; i++) {
		const char = line[i];

		if (insideQuotes) {
			if (char === '"') {
				if (line[i + 1] === '"') {
					currentValue += '"';
					i++;
				} else {
					insideQuotes = false;
				}
			} else {
				currentValue += char;
			}
			continue;
		}

		if (char === ';') {
			values.push(currentValue);
			currentValue = '';
			continue;
		}

		if (char === '"' && currentValue === '') {
			insideQuotes = true;
			continue;
		}

		currentValue += char;
	}

	values.push(currentValue);
	return values;
}

function escapeCsvValue(value) {
	if (value === '') return '';

	if (/[;"\r\n]/.test(value)) {
		return `"${value.replace(/"/g, '""')}"`;
	}

	return value;
}

function processFile(fileName) {
	const filePath = path.join(inputDirectory, fileName);

	const content = fs.readFileSync(filePath, 'utf8').replace(/^\uFEFF/, '');
	const lines = content.split(/\r?\n/);

	if (lines.length === 0) return false;

	const header = splitCsvLine(lines[0]);
	const dataIndex = header.findIndex(
		(column) => column.trim().toLowerCase() === 'data_inversa'
	);

	if (dataIndex === -1) {
		console.log(`${fileName}: coluna data_inversa não encontrada.`);
		return false;
	}

	let modified = false;
	const outputLines = [lines[0]];

	for (let i = 1; i < lines.length; i++) {
		if (lines[i].trim() === '') {
			outputLines.push(lines[i]);
			continue;
		}

		const values = splitCsvLine(lines[i]);

		if (values.length > dataIndex) {
			const value = values[dataIndex].trim();

			if (/^\d{2}\/\d{2}\/\d{4}$/.test(value)) {
				const [day, month, year] = value.split('/');
				values[dataIndex] = `${year}-${month}-${day}`;
				modified = true;
			}
		}

		outputLines.push(values.map(escapeCsvValue).join(';'));
	}

	if (modified) {
		fs.writeFileSync(filePath, outputLines.join('\n'), 'utf8');
		console.log(`✔ ${fileName} corrigido.`);
	} else {
		console.log(`- ${fileName} já estava correto.`);
	}

	return modified;
}

function main() {
	const files = findInputFiles(inputDirectory);

	if (files.length === 0) {
		throw new Error('Nenhum arquivo encontrado.');
	}

	let total = 0;

	for (const file of files) {
		if (processFile(file)) {
			total++;
		}
	}

	console.log(`\n${total} arquivo(s) alterado(s).`);
}

try {
	main();
} catch (err) {
	console.error(err.message);
	process.exit(1);
}